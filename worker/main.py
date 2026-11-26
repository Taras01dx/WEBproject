"""Independent Compute Worker Application Server (FastAPI)."""

from __future__ import annotations

import asyncio
import os
import resource
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from shared.config import BALANCER_INTERNAL_URL, INTERNAL_API_KEY, WORKER_MAX_CONCURRENCY
from worker.solver import (
    SingularMatrixError,
    TaskCancelledError,
    TaskTimeoutError,
    solve_slae,
)

WORKER_ID = os.getenv("WORKER_ID", "worker-1")
MAX_CONCURRENCY = int(os.getenv("WORKER_MAX_CONCURRENCY", str(WORKER_MAX_CONCURRENCY)))
BALANCER_URL = os.getenv("BALANCER_INTERNAL_URL", BALANCER_INTERNAL_URL).rstrip("/")
STARTED_AT = time.time()

app = FastAPI(title=f"SLAE Compute Worker ({WORKER_ID})", version="1.0.0")

# Dedicated compute pool; NumPy releases the GIL during linear algebra operations,
# and each worker runs in its own OS process / container for true multi-core parallelism.
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY, thread_name_prefix=f"{WORKER_ID}-compute")

_state_lock = threading.Lock()
_active_tasks: dict[str, dict[str, Any]] = {}
_cancel_events: dict[str, threading.Event] = {}
_total_completed = 0
_total_failed = 0
_total_cancelled = 0


class ExecuteRequest(BaseModel):
    task_id: str
    matrix_size: int
    method: str = "gauss"
    input_mode: str = "random"
    seed: int | None = None
    custom_payload: dict[str, Any] | None = None
    timeout_sec: int = 45


def _verify_internal_key(x_internal_key: str | None) -> None:
    if x_internal_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden internal request")


def _get_memory_mb() -> float:
    """Read resident set size (RSS) in MB in a cross-platform way."""
    try:
        # On Linux, ru_maxrss is in kilobytes
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss_kb / 1024.0, 1)
    except Exception:
        return 0.0


def _get_cpu_percent() -> float:
    """Estimate current worker CPU utilization from active slots and OS load average."""
    with _state_lock:
        slot_ratio = len(_active_tasks) / max(1, MAX_CONCURRENCY)
    try:
        load1 = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        os_load = min(100.0, (load1 / cores) * 100.0)
        return round(max(slot_ratio * 85.0, min(100.0, os_load)), 1)
    except Exception:
        return round(slot_ratio * 90.0, 1)


def _send_callback_sync(path: str, payload: dict[str, Any]) -> None:
    """Send synchronous HTTP callback from compute thread to Load Balancer."""
    url = f"{BALANCER_URL}{path}"
    headers = {"X-Internal-Key": INTERNAL_API_KEY}
    try:
        with httpx.Client(timeout=5.0, verify=False) as client:
            client.post(url, json=payload, headers=headers)
    except Exception:
        pass


def _run_task_in_thread(req: ExecuteRequest, cancel_event: threading.Event) -> None:
    global _total_completed, _total_failed, _total_cancelled
    last_sent = 0.0

    def on_progress(progress: float, stage: str) -> None:
        nonlocal last_sent
        now = time.monotonic()
        with _state_lock:
            if req.task_id in _active_tasks:
                _active_tasks[req.task_id]["progress"] = progress
                _active_tasks[req.task_id]["stage"] = stage
        # Throttle network callbacks while always delivering start & completion milestones
        if progress in (2.0, 10.0, 100.0) or (now - last_sent) >= 0.35:
            last_sent = now
            _send_callback_sync(
                f"/internal/tasks/{req.task_id}/progress",
                {
                    "worker_id": WORKER_ID,
                    "progress": progress,
                    "stage": stage,
                },
            )

    try:
        result = solve_slae(
            matrix_size=req.matrix_size,
            method=req.method,
            input_mode=req.input_mode,
            seed=req.seed,
            custom_payload=req.custom_payload,
            timeout_sec=float(req.timeout_sec),
            cancel_event=cancel_event,
            on_progress=on_progress,
        )
        with _state_lock:
            _total_completed += 1
        _send_callback_sync(
            f"/internal/tasks/{req.task_id}/complete",
            {
                "worker_id": WORKER_ID,
                "status": "completed",
                "progress": 100.0,
                "stage": "Завершено",
                **result,
            },
        )
    except TaskCancelledError as exc:
        with _state_lock:
            _total_cancelled += 1
        _send_callback_sync(
            f"/internal/tasks/{req.task_id}/complete",
            {
                "worker_id": WORKER_ID,
                "status": "cancelled",
                "stage": "Скасовано",
                "error_message": str(exc),
            },
        )
    except (TaskTimeoutError, SingularMatrixError, ValueError) as exc:
        with _state_lock:
            _total_failed += 1
        _send_callback_sync(
            f"/internal/tasks/{req.task_id}/complete",
            {
                "worker_id": WORKER_ID,
                "status": "failed",
                "stage": "Помилка виконання",
                "error_message": str(exc),
            },
        )
    except Exception as exc:
        with _state_lock:
            _total_failed += 1
        _send_callback_sync(
            f"/internal/tasks/{req.task_id}/complete",
            {
                "worker_id": WORKER_ID,
                "status": "failed",
                "stage": "Внутрішня помилка воркера",
                "error_message": f"Помилка обчислення: {exc}",
            },
        )
    finally:
        with _state_lock:
            _active_tasks.pop(req.task_id, None)
            _cancel_events.pop(req.task_id, None)


@app.get("/health")
async def health() -> dict[str, Any]:
    with _state_lock:
        active_count = len(_active_tasks)
    return {
        "status": "ok",
        "worker_id": WORKER_ID,
        "active_tasks": active_count,
        "max_concurrency": MAX_CONCURRENCY,
        "available_slots": max(0, MAX_CONCURRENCY - active_count),
        "uptime_sec": round(time.time() - STARTED_AT, 1),
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    with _state_lock:
        active_list = list(_active_tasks.values())
        active_count = len(active_list)
        completed = _total_completed
        failed = _total_failed
        cancelled = _total_cancelled

    return {
        "worker_id": WORKER_ID,
        "pid": os.getpid(),
        "active_tasks": active_count,
        "max_concurrency": MAX_CONCURRENCY,
        "utilization": round(active_count / max(1, MAX_CONCURRENCY), 2),
        "cpu_percent": _get_cpu_percent(),
        "memory_mb": _get_memory_mb(),
        "uptime_sec": round(time.time() - STARTED_AT, 1),
        "completed_total": completed,
        "failed_total": failed,
        "cancelled_total": cancelled,
        "tasks": active_list,
    }


@app.post("/execute")
async def execute_task(
    req: ExecuteRequest,
    x_internal_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_internal_key(x_internal_key)

    with _state_lock:
        if len(_active_tasks) >= MAX_CONCURRENCY:
            raise HTTPException(status_code=503, detail="Worker capacity reached")
        cancel_event = threading.Event()
        _cancel_events[req.task_id] = cancel_event
        _active_tasks[req.task_id] = {
            "task_id": req.task_id,
            "matrix_size": req.matrix_size,
            "method": req.method,
            "progress": 0.0,
            "stage": "Прийнято воркером",
            "started_at": time.time(),
        }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, _run_task_in_thread, req, cancel_event)
    return {"accepted": True, "worker_id": WORKER_ID, "task_id": req.task_id}


@app.post("/cancel/{task_id}")
async def cancel_task(
    task_id: str,
    x_internal_key: str | None = Header(default=None),
) -> dict[str, Any]:
    _verify_internal_key(x_internal_key)
    with _state_lock:
        ev = _cancel_events.get(task_id)
        if not ev:
            return {"cancelled": False, "reason": "not_running_on_worker"}
        ev.set()
    return {"cancelled": True, "worker_id": WORKER_ID, "task_id": task_id}
