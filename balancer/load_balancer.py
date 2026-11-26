"""Least-Connections Load Balancer, FIFO Queue Manager, ETA Estimator, and SSE Broadcaster."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from shared.config import (
    HEARTBEAT_INTERVAL_SEC,
    INTERNAL_API_KEY,
    MAX_QUEUE_SIZE,
    WORKER_MAX_CONCURRENCY,
    WORKER_TIMEOUT_SEC,
)
from shared.database import Database


@dataclass
class WorkerNode:
    worker_id: str
    url: str
    max_concurrency: int = WORKER_MAX_CONCURRENCY
    is_dynamic: bool = False
    status: str = "starting"  # starting | online | offline
    active_tasks: int = 0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    uptime_sec: float = 0.0
    completed_total: int = 0
    failed_total: int = 0
    cancelled_total: int = 0
    pid: int | None = None
    container_id: str | None = None
    last_heartbeat: float = 0.0
    idle_since: float = field(default_factory=time.time)
    last_dispatched_at: float = 0.0

    @property
    def available_slots(self) -> int:
        if self.status != "online":
            return 0
        return max(0, self.max_concurrency - self.active_tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "url": self.url,
            "status": self.status,
            "is_dynamic": self.is_dynamic,
            "active_tasks": self.active_tasks,
            "max_concurrency": self.max_concurrency,
            "available_slots": self.available_slots,
            "utilization": round(self.active_tasks / max(1, self.max_concurrency), 2),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "uptime_sec": self.uptime_sec,
            "completed_total": self.completed_total,
            "failed_total": self.failed_total,
            "cancelled_total": self.cancelled_total,
            "pid": self.pid,
            "last_heartbeat": self.last_heartbeat,
        }


class LoadBalancer:
    """Manages worker pool, least-connections routing, persistent FIFO queue, ETA, and SSE updates."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.workers: dict[str, WorkerNode] = {}
        self._dispatch_lock = asyncio.Lock()
        self._rr_counter = 0
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._running = False
        self._poll_task: asyncio.Task[Any] | None = None

    # ---------------- Worker Registry ----------------

    def register_worker(
        self,
        worker_id: str,
        url: str,
        max_concurrency: int = WORKER_MAX_CONCURRENCY,
        is_dynamic: bool = False,
        container_id: str | None = None,
    ) -> WorkerNode:
        node = WorkerNode(
            worker_id=worker_id,
            url=url.rstrip("/"),
            max_concurrency=max_concurrency,
            is_dynamic=is_dynamic,
            container_id=container_id,
            status="starting",
            idle_since=time.time(),
        )
        self.workers[worker_id] = node
        return node

    def unregister_worker(self, worker_id: str) -> None:
        self.workers.pop(worker_id, None)

    def list_workers(self) -> list[dict[str, Any]]:
        return [w.to_dict() for w in sorted(self.workers.values(), key=lambda x: x.worker_id)]

    def get_cluster_capacity(self) -> dict[str, Any]:
        online = [w for w in self.workers.values() if w.status == "online"]
        total_slots = sum(w.max_concurrency for w in online)
        used_slots = sum(w.active_tasks for w in online)
        queued = len(self.db.list_queued_tasks())
        utilization = round(used_slots / max(1, total_slots), 3) if total_slots > 0 else 0.0
        return {
            "total_workers": len(self.workers),
            "online_workers": len(online),
            "dynamic_workers": sum(1 for w in self.workers.values() if w.is_dynamic),
            "total_slots": total_slots,
            "used_slots": used_slots,
            "available_slots": max(0, total_slots - used_slots),
            "utilization": utilization,
            "queue_length": queued,
            "max_queue_size": MAX_QUEUE_SIZE,
        }

    # ---------------- Health & Metrics Polling ----------------

    async def start(self) -> None:
        self._running = True
        await self.poll_workers_once()
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.poll_workers_once()
                await self.dispatch_queued_tasks()
            except Exception:
                pass
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)

    async def poll_workers_once(self) -> None:
        now = time.time()
        async with httpx.AsyncClient(timeout=3.0) as client:
            for worker in list(self.workers.values()):
                try:
                    resp = await client.get(f"{worker.url}/metrics")
                    if resp.status_code == 200:
                        data = resp.json()
                        prev_status = worker.status
                        worker.status = "online"
                        worker.active_tasks = int(data.get("active_tasks", 0))
                        worker.max_concurrency = int(data.get("max_concurrency", worker.max_concurrency))
                        worker.cpu_percent = float(data.get("cpu_percent", 0.0))
                        worker.memory_mb = float(data.get("memory_mb", 0.0))
                        worker.uptime_sec = float(data.get("uptime_sec", 0.0))
                        worker.completed_total = int(data.get("completed_total", 0))
                        worker.failed_total = int(data.get("failed_total", 0))
                        worker.cancelled_total = int(data.get("cancelled_total", 0))
                        worker.pid = data.get("pid")
                        worker.last_heartbeat = now
                        if worker.active_tasks == 0:
                            if prev_status != "online" or worker.idle_since == 0.0:
                                worker.idle_since = now
                        else:
                            worker.idle_since = now
                    else:
                        self._mark_worker_unhealthy(worker, now)
                except Exception:
                    self._mark_worker_unhealthy(worker, now)

    @staticmethod
    def _mark_worker_unhealthy(worker: WorkerNode, now: float) -> None:
        if worker.last_heartbeat == 0.0 or (now - worker.last_heartbeat) > WORKER_TIMEOUT_SEC:
            worker.status = "offline"

    # ---------------- Least-Connections Selection ----------------

    def _select_best_worker(self) -> WorkerNode | None:
        """Select healthy worker with lowest utilization (Least-Connections + Round-Robin tie-break)."""
        candidates = [
            w
            for w in self.workers.values()
            if w.status == "online" and w.active_tasks < w.max_concurrency
        ]
        if not candidates:
            return None

        # Sort by (utilization ratio, active_tasks, last_dispatched_at) for fair 50/50 distribution
        candidates.sort(
            key=lambda w: (
                w.active_tasks / max(1, w.max_concurrency),
                w.active_tasks,
                w.last_dispatched_at,
                w.worker_id,
            )
        )
        return candidates[0]

    # ---------------- ETA Estimation (Requirement 6) ----------------

    def estimate_task_duration_sec(self, matrix_size: int, method: str) -> float:
        """Predict task runtime in seconds using empirical DB calibration or O(n^3)/O(n^2) baseline."""
        calibrated_k = self.db.get_average_ms_per_cubic_n(method)
        if method == "gauss":
            # Default empirical constant for Python/NumPy partial pivoting (~1.5e-6 ms * n^3)
            k = calibrated_k if calibrated_k is not None else 1.8e-6
            est_ms = k * (matrix_size**3)
        else:
            k = calibrated_k if calibrated_k is not None else 2.5e-3
            est_ms = k * (matrix_size**2)
        return max(0.8, round(est_ms / 1000.0, 1))

    def _update_queue_positions_and_eta(self) -> None:
        """Recalculate queue position and estimated wait time for all queued tasks."""
        queued_tasks = self.db.list_queued_tasks()
        if not queued_tasks:
            return

        active_tasks = [t for t in self.db.list_active_tasks() if t["status"] == "running"]
        cap = self.get_cluster_capacity()
        total_slots = max(1, cap["total_slots"])

        # Remaining work across currently running tasks
        now = time.time()
        running_remaining: list[float] = []
        for rt in active_tasks:
            prog = max(1.0, float(rt.get("progress", 0.0)))
            started = rt.get("started_at") or now
            elapsed = max(0.2, now - float(started))
            if prog >= 10.0:
                est_rem = elapsed * ((100.0 - prog) / prog)
            else:
                est_rem = self.estimate_task_duration_sec(int(rt["matrix_size"]), str(rt["method"]))
            running_remaining.append(max(0.5, min(float(rt.get("timeout_sec", 45)), est_rem)))

        min_running_finish = min(running_remaining) if running_remaining else 1.0
        cumulative_ahead_sec = sum(running_remaining)

        for idx, qt in enumerate(queued_tasks, start=1):
            if idx == 1:
                eta = max(1.0, round(min_running_finish, 1))
            else:
                eta = max(1.0, round(cumulative_ahead_sec / total_slots, 1))

            stage_msg = f"У черзі (позиція #{idx}, очікування ~{int(round(eta))} с)"
            updated = self.db.update_task(
                qt["id"],
                queue_position=idx,
                estimated_wait_sec=eta,
                stage=stage_msg,
            )
            if updated:
                self.publish_task_update(updated)

            cumulative_ahead_sec += self.estimate_task_duration_sec(
                int(qt["matrix_size"]), str(qt["method"])
            )

    # ---------------- Task Submission & Dispatch ----------------

    async def submit_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Try immediate dispatch to best worker; otherwise enqueue with position & ETA."""
        async with self._dispatch_lock:
            worker = self._select_best_worker()
            if worker is not None:
                dispatched = await self._dispatch_to_worker(task, worker)
                if dispatched is not None:
                    return dispatched

            # All workers busy (or offline): keep in FIFO queue and compute ETA
            self._update_queue_positions_and_eta()
            refreshed = self.db.get_task(task["id"]) or task
            self.db.log_event(
                "QUEUE_ENQUEUE",
                "load-balancer",
                f"Усі слоти зайняті — задачу {task['id'][:8]} поставлено у чергу (позиція #{refreshed.get('queue_position', 1)}, ETA ~{refreshed.get('estimated_wait_sec', 0)} с)",
                {"task_id": task["id"], "queue_position": refreshed.get("queue_position")},
            )
            return refreshed

    async def dispatch_queued_tasks(self) -> None:
        """Drain queued tasks whenever worker slots become free."""
        async with self._dispatch_lock:
            queued = self.db.list_queued_tasks()
            if not queued:
                return
            for task in queued:
                if task.get("cancel_requested"):
                    continue
                worker = self._select_best_worker()
                if worker is None:
                    break
                await self._dispatch_to_worker(task, worker)
            self._update_queue_positions_and_eta()

    async def _dispatch_to_worker(self, task: dict[str, Any], worker: WorkerNode) -> dict[str, Any] | None:
        payload = {
            "task_id": task["id"],
            "matrix_size": task["matrix_size"],
            "method": task["method"],
            "input_mode": task["input_mode"],
            "seed": task["seed"],
            "custom_payload": task["custom_payload"],
            "timeout_sec": task["timeout_sec"],
        }
        headers = {"X-Internal-Key": INTERNAL_API_KEY}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{worker.url}/execute", json=payload, headers=headers)
            if resp.status_code == 200:
                worker.active_tasks += 1
                worker.last_dispatched_at = time.time()
                worker.idle_since = time.time()
                updated = self.db.update_task(
                    task["id"],
                    status="running",
                    worker_id=worker.worker_id,
                    queue_position=0,
                    estimated_wait_sec=0.0,
                    started_at=time.time(),
                    progress=1.0,
                    stage=f"Призначено на сервер {worker.worker_id}",
                )
                if updated:
                    self.publish_task_update(updated)
                return updated
            if resp.status_code == 503:
                worker.active_tasks = worker.max_concurrency
        except Exception:
            worker.status = "offline"
        return None

    # ---------------- Cancellation & Worker Callbacks ----------------

    async def cancel_task(self, task_id: str, actor: str = "user") -> dict[str, Any] | None:
        task = self.db.get_task(task_id)
        if not task:
            return None
        if task["status"] in ("completed", "failed", "cancelled"):
            return task

        if task["status"] == "queued":
            updated = self.db.update_task(
                task_id,
                status="cancelled",
                cancel_requested=1,
                queue_position=0,
                estimated_wait_sec=0.0,
                stage="Скасовано з черги",
                error_message=f"Задачу скасовано ({actor}) до початку виконання.",
                finished_at=time.time(),
            )
            self._update_queue_positions_and_eta()
            if updated:
                self.publish_task_update(updated)
            self.db.log_event("TASK_CANCELLED", actor, f"Скасовано задачу {task_id[:8]} із черги")
            return updated

        # Task is running on a worker -> send cancellation signal
        self.db.update_task(task_id, cancel_requested=1, stage="Надсилання запиту на скасування...")
        worker_id = task.get("worker_id")
        worker = self.workers.get(worker_id) if worker_id else None
        if worker and worker.status == "online":
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    await client.post(
                        f"{worker.url}/cancel/{task_id}",
                        headers={"X-Internal-Key": INTERNAL_API_KEY},
                    )
            except Exception:
                pass
        updated = self.db.get_task(task_id)
        if updated:
            self.publish_task_update(updated)
        self.db.log_event("TASK_CANCELLED", actor, f"Запит на скасування задачі {task_id[:8]} на {worker_id}")
        return updated

    async def handle_worker_progress(self, task_id: str, worker_id: str, progress: float, stage: str) -> None:
        task = self.db.get_task(task_id)
        if not task or task["status"] in ("completed", "failed", "cancelled"):
            return
        updated = self.db.update_task(task_id, progress=progress, stage=stage, worker_id=worker_id)
        if updated:
            self.publish_task_update(updated)

    async def handle_worker_complete(self, task_id: str, payload: dict[str, Any]) -> None:
        worker_id = payload.get("worker_id")
        if worker_id and worker_id in self.workers:
            w = self.workers[worker_id]
            w.active_tasks = max(0, w.active_tasks - 1)
            if w.active_tasks == 0:
                w.idle_since = time.time()

        status = payload.get("status", "completed")
        fields: dict[str, Any] = {
            "status": status,
            "stage": payload.get("stage", "Завершено"),
            "finished_at": time.time(),
        }
        if status == "completed":
            fields["progress"] = 100.0
            fields["execution_time_ms"] = payload.get("execution_time_ms")
            fields["residual_norm"] = payload.get("residual_norm")
            fields["iterations"] = payload.get("iterations")
            fields["solution_preview"] = payload.get("solution_preview")
        else:
            fields["error_message"] = payload.get("error_message")

        updated = self.db.update_task(task_id, **fields)
        if updated:
            self.publish_task_update(updated)

        # Immediately dispatch next queued task to the freed slot!
        asyncio.create_task(self.dispatch_queued_tasks())

    # ---------------- Real-time SSE Pub/Sub ----------------

    def subscribe(self, task_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=32)
        self._subscribers.setdefault(task_id, set()).add(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._subscribers.get(task_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subscribers.pop(task_id, None)

    def publish_task_update(self, task: dict[str, Any]) -> None:
        subs = self._subscribers.get(task["id"])
        if not subs:
            return
        for q in list(subs):
            try:
                q.put_nowait(task)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(task)
                except Exception:
                    pass
