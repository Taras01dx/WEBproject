"""Horizontal Autoscaler (Requirement 7): spawns/stops worker servers within [MIN_WORKERS, MAX_WORKERS]."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from typing import Any

import httpx

from balancer.load_balancer import LoadBalancer
from shared.config import (
    AUTOSCALE_CHECK_INTERVAL_SEC,
    AUTOSCALE_ENABLED,
    AUTOSCALE_MODE,
    BALANCER_INTERNAL_URL,
    BASE_DIR,
    DOCKER_NETWORK_NAME,
    DOCKER_SOCKET_PATH,
    DOCKER_WORKER_IMAGE,
    DYNAMIC_WORKER_PORT_START,
    INTERNAL_API_KEY,
    MAX_WORKERS,
    MIN_WORKERS,
    SCALE_DOWN_IDLE_SEC,
    SCALE_UP_COOLDOWN_SEC,
    SCALE_UP_UTILIZATION_THRESHOLD,
    WORKER_MAX_CONCURRENCY,
)
from shared.database import Database


class HorizontalAutoscaler:
    """Automatically scales compute workers up under load and down when idle."""

    def __init__(self, lb: LoadBalancer, db: Database) -> None:
        self.lb = lb
        self.db = db
        self.enabled = AUTOSCALE_ENABLED
        self.mode = AUTOSCALE_MODE  # "local" or "docker"
        self.min_workers = MIN_WORKERS
        self.max_workers = MAX_WORKERS
        self._running = False
        self._task: asyncio.Task[Any] | None = None
        self._last_scale_up = 0.0
        self._scaling_lock = asyncio.Lock()
        self._local_procs: dict[str, subprocess.Popen[Any]] = {}
        self._used_ports: set[int] = set()

    def status_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "current_workers": len(self.lb.workers),
            "dynamic_workers": sum(1 for w in self.lb.workers.values() if w.is_dynamic),
            "scale_up_threshold": SCALE_UP_UTILIZATION_THRESHOLD,
            "scale_down_idle_sec": SCALE_DOWN_IDLE_SEC,
        }

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._control_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Cleanly terminate any dynamically spawned local workers
        for wid in list(self._local_procs.keys()):
            await self._stop_local_worker(wid)

    async def _control_loop(self) -> None:
        while self._running:
            try:
                if self.enabled:
                    await self._evaluate()
            except Exception:
                pass
            await asyncio.sleep(AUTOSCALE_CHECK_INTERVAL_SEC)

    async def _evaluate(self) -> None:
        cap = self.lb.get_cluster_capacity()
        now = time.time()

        # Scale UP when cluster is heavily utilized and queue has waiting tasks
        if (
            cap["total_workers"] < self.max_workers
            and cap["queue_length"] >= 1
            and cap["utilization"] >= SCALE_UP_UTILIZATION_THRESHOLD
            and (now - self._last_scale_up) >= SCALE_UP_COOLDOWN_SEC
        ):
            await self.scale_up(reason=f"Автоскейлінг: завантаження {int(cap['utilization']*100)}%, у черзі {cap['queue_length']} задач")
            return

        # Scale DOWN idle dynamic workers when queue is empty
        if cap["queue_length"] == 0 and cap["total_workers"] > self.min_workers:
            dynamic_idle = [
                w
                for w in self.lb.workers.values()
                if w.is_dynamic
                and w.active_tasks == 0
                and (now - w.idle_since) >= SCALE_DOWN_IDLE_SEC
            ]
            if dynamic_idle:
                # Terminate the most recently idle extra worker
                victim = sorted(dynamic_idle, key=lambda w: w.worker_id, reverse=True)[0]
                await self.scale_down(
                    worker_id=victim.worker_id,
                    reason=f"Автоскейлінг: воркер {victim.worker_id} простоював > {int(SCALE_DOWN_IDLE_SEC)} с",
                )

    def _next_worker_id_and_port(self) -> tuple[str, int]:
        idx = self.min_workers + 1
        while f"worker-{idx}" in self.lb.workers:
            idx += 1
        port = DYNAMIC_WORKER_PORT_START
        while port in self._used_ports:
            port += 1
        return f"worker-{idx}", port

    async def scale_up(self, reason: str = "Ручний запуск") -> dict[str, Any]:
        async with self._scaling_lock:
            if len(self.lb.workers) >= self.max_workers:
                return {
                    "scaled": False,
                    "error": f"Досягнуто максимальну кількість серверів ({self.max_workers}).",
                }

            worker_id, port = self._next_worker_id_and_port()
            self._last_scale_up = time.time()

            if self.mode == "docker" and os.path.exists(DOCKER_SOCKET_PATH):
                node = await self._spawn_docker_worker(worker_id)
            else:
                node = await self._spawn_local_worker(worker_id, port)

            if node is None:
                return {"scaled": False, "error": f"Не вдалося запустити сервер {worker_id}."}

            await self.lb.poll_workers_once()
            await self.lb.dispatch_queued_tasks()
            self.db.log_event(
                "SCALE_UP",
                "autoscaler",
                f"Запущено новий сервер застосувань {worker_id} ({node.url}). {reason}",
                {"worker_id": worker_id, "mode": self.mode, "total_workers": len(self.lb.workers)},
            )
            return {"scaled": True, "worker": node.to_dict()}

    async def scale_down(self, worker_id: str | None = None, reason: str = "Ручна зупинка") -> dict[str, Any]:
        async with self._scaling_lock:
            if len(self.lb.workers) <= self.min_workers:
                return {
                    "scaled": False,
                    "error": f"Неможливо зменшити кількість серверів нижче мінімуму ({self.min_workers}).",
                }

            target_id = worker_id
            if not target_id:
                dynamic_idle = [
                    w for w in self.lb.workers.values() if w.is_dynamic and w.active_tasks == 0
                ]
                if not dynamic_idle:
                    return {
                        "scaled": False,
                        "error": "Немає вільних динамічних серверів для зупинки.",
                    }
                target_id = sorted(dynamic_idle, key=lambda w: w.worker_id, reverse=True)[0].worker_id

            node = self.lb.workers.get(target_id)
            if not node or not node.is_dynamic:
                return {"scaled": False, "error": "Можна зупиняти лише динамічно створені сервери."}
            if node.active_tasks > 0:
                return {"scaled": False, "error": f"Сервер {target_id} зараз виконує задачі."}

            if node.container_id and self.mode == "docker":
                await self._stop_docker_worker(target_id, node.container_id)
            else:
                await self._stop_local_worker(target_id)

            self.lb.unregister_worker(target_id)
            self.db.log_event(
                "SCALE_DOWN",
                "autoscaler",
                f"Зупинено динамічний сервер {target_id}. {reason}",
                {"worker_id": target_id, "total_workers": len(self.lb.workers)},
            )
            return {"scaled": True, "removed_worker_id": target_id}

    # ---------------- Local Subprocess Spawner ----------------

    async def _spawn_local_worker(self, worker_id: str, port: int) -> Any:
        url = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env.update(
            {
                "WORKER_ID": worker_id,
                "WORKER_MAX_CONCURRENCY": str(WORKER_MAX_CONCURRENCY),
                "BALANCER_INTERNAL_URL": BALANCER_INTERNAL_URL,
                "INTERNAL_API_KEY": INTERNAL_API_KEY,
                "PYTHONPATH": str(BASE_DIR),
            }
        )
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "worker.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._local_procs[worker_id] = proc
        self._used_ports.add(port)

        ready = await self._wait_for_health(url, timeout_sec=10.0)
        if not ready:
            await self._stop_local_worker(worker_id)
            return None

        node = self.lb.register_worker(
            worker_id=worker_id,
            url=url,
            max_concurrency=WORKER_MAX_CONCURRENCY,
            is_dynamic=True,
        )
        node.status = "online"
        node.last_heartbeat = time.time()
        return node

    async def _stop_local_worker(self, worker_id: str) -> None:
        proc = self._local_procs.pop(worker_id, None)
        node = self.lb.workers.get(worker_id)
        if node and ":" in node.url:
            try:
                port = int(node.url.rsplit(":", 1)[1])
                self._used_ports.discard(port)
            except Exception:
                pass
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # ---------------- Docker Engine Unix Socket Spawner ----------------

    async def _spawn_docker_worker(self, worker_id: str) -> Any:
        """Spawn a new worker container via Docker Engine API over /var/run/docker.sock."""
        container_name = f"slae-{worker_id}"
        transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET_PATH)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=15.0) as client:
                create_body = {
                    "Image": DOCKER_WORKER_IMAGE,
                    "Cmd": ["uvicorn", "worker.main:app", "--host", "0.0.0.0", "--port", "8001"],
                    "Env": [
                        f"WORKER_ID={worker_id}",
                        f"WORKER_MAX_CONCURRENCY={WORKER_MAX_CONCURRENCY}",
                        f"BALANCER_INTERNAL_URL={BALANCER_INTERNAL_URL}",
                        f"INTERNAL_API_KEY={INTERNAL_API_KEY}",
                    ],
                    "HostConfig": {
                        "NetworkMode": DOCKER_NETWORK_NAME,
                        "AutoRemove": True,
                    },
                    "NetworkingConfig": {
                        "EndpointsConfig": {
                            DOCKER_NETWORK_NAME: {"Aliases": [worker_id]}
                        }
                    },
                }
                resp = await client.post(f"/containers/create?name={container_name}", json=create_body)
                if resp.status_code not in (200, 201):
                    return None
                container_id = resp.json()["Id"]
                start_resp = await client.post(f"/containers/{container_id}/start")
                if start_resp.status_code not in (200, 204):
                    return None
        except Exception:
            return None

        url = f"http://{worker_id}:8001"
        ready = await self._wait_for_health(url, timeout_sec=15.0)
        if not ready:
            await self._stop_docker_worker(worker_id, container_id)
            return None

        node = self.lb.register_worker(
            worker_id=worker_id,
            url=url,
            max_concurrency=WORKER_MAX_CONCURRENCY,
            is_dynamic=True,
            container_id=container_id,
        )
        node.status = "online"
        node.last_heartbeat = time.time()
        return node

    async def _stop_docker_worker(self, worker_id: str, container_id: str) -> None:
        transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET_PATH)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=10.0) as client:
                await client.post(f"/containers/{container_id}/stop?t=3")
        except Exception:
            pass

    @staticmethod
    async def _wait_for_health(url: str, timeout_sec: float = 10.0) -> bool:
        deadline = time.time() + timeout_sec
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                try:
                    resp = await client.get(f"{url}/health")
                    if resp.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.35)
        return False
