"""Gateway & Load Balancer FastAPI Server — Auth, Task API, SSE Progress, Admin Panel & Static UI."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from balancer.autoscaler import HorizontalAutoscaler
from balancer.load_balancer import LoadBalancer
from shared.config import (
    ALLOWED_METHODS,
    BASE_DIR,
    INTERNAL_API_KEY,
    MAX_ACTIVE_TASKS_PER_USER,
    MAX_CUSTOM_MATRIX_SIZE,
    MAX_MATRIX_SIZE,
    MAX_QUEUE_SIZE,
    MAX_TIMEOUT_SEC,
    MAX_TOTAL_TASKS_PER_USER,
    MAX_WORKERS,
    MIN_MATRIX_SIZE,
    MIN_TIMEOUT_SEC,
    MIN_WORKERS,
    WORKER_MAX_CONCURRENCY,
    parse_initial_workers,
)
from shared.database import Database
from shared.security import create_access_token, decode_access_token, verify_password

db = Database()
lb = LoadBalancer(db)
autoscaler = HorizontalAutoscaler(lb, db)

app = FastAPI(title="SLAE Compute Cluster — Load Balancer & Gateway", version="1.0.0")

FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------- Lifecycle ----------------


@app.on_event("startup")
async def on_startup() -> None:
    db.mark_orphaned_tasks_failed()
    for wid, url in parse_initial_workers():
        lb.register_worker(wid, url, max_concurrency=WORKER_MAX_CONCURRENCY, is_dynamic=False)
    await lb.start()
    await autoscaler.start()
    db.log_event(
        "CLUSTER_START",
        "load-balancer",
        f"Кластер запущено з {len(lb.workers)} базовими серверами застосувань (мін={MIN_WORKERS}, макс={MAX_WORKERS})",
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await autoscaler.stop()
    await lb.stop()


# ---------------- Pydantic Schemas ----------------


class AuthPayload(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=128)


class CustomSystemPayload(BaseModel):
    matrix: list[list[float]]
    vector: list[float]


class CreateTaskPayload(BaseModel):
    matrix_size: int
    method: str = "gauss"
    input_mode: str = "random"  # "random" | "custom"
    seed: int | None = None
    timeout_sec: int = MAX_TIMEOUT_SEC
    custom_payload: CustomSystemPayload | None = None


class WorkerProgressPayload(BaseModel):
    worker_id: str
    progress: float
    stage: str


class WorkerCompletePayload(BaseModel):
    worker_id: str
    status: str
    progress: float = 100.0
    stage: str = "Завершено"
    execution_time_ms: float | None = None
    residual_norm: float | None = None
    iterations: int | None = None
    solution_preview: list[float] | None = None
    error_message: str | None = None


# ---------------- Auth Dependencies ----------------


def _extract_token(request: Request, query_token: str | None = None) -> str | None:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        return cookie_token
    if query_token:
        return query_token
    return None


def get_current_user(
    request: Request,
    token: str | None = Query(default=None),
) -> dict[str, Any]:
    raw_token = _extract_token(request, token)
    if not raw_token:
        raise HTTPException(status_code=401, detail="Потрібна авторизація.")
    payload = decode_access_token(raw_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Недійсний або протермінований токен.")
    user = db.get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="Користувача не знайдено.")
    return user


def require_admin(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Доступ лише для адміністратора.")
    return user


def _verify_internal_key(x_internal_key: str | None) -> None:
    if x_internal_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden internal callback")


# ---------------- Frontend Entry ----------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "role": "load-balancer",
        "capacity": lb.get_cluster_capacity(),
        "autoscaler": autoscaler.status_dict(),
    }


@app.get("/api/limits")
async def get_limits() -> dict[str, Any]:
    return {
        "min_matrix_size": MIN_MATRIX_SIZE,
        "max_matrix_size": MAX_MATRIX_SIZE,
        "max_custom_matrix_size": MAX_CUSTOM_MATRIX_SIZE,
        "min_timeout_sec": MIN_TIMEOUT_SEC,
        "max_timeout_sec": MAX_TIMEOUT_SEC,
        "allowed_methods": list(ALLOWED_METHODS),
        "max_active_tasks_per_user": MAX_ACTIVE_TASKS_PER_USER,
        "max_total_tasks_per_user": MAX_TOTAL_TASKS_PER_USER,
        "max_queue_size": MAX_QUEUE_SIZE,
        "min_workers": MIN_WORKERS,
        "max_workers": MAX_WORKERS,
        "worker_max_concurrency": WORKER_MAX_CONCURRENCY,
    }


# ---------------- Auth Endpoints (Requirement 4) ----------------


@app.post("/api/auth/register")
async def register(payload: AuthPayload, response: Response) -> dict[str, Any]:
    username = payload.username.strip()
    if not username.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(
            status_code=400,
            detail="Ім'я користувача може містити лише літери, цифри, '_' та '-'.",
        )
    try:
        user = db.create_user(username=username, password=payload.password, role="user")
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Користувач із таким іменем вже існує.")

    token = create_access_token(user["id"], user["username"], user["role"])
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=24 * 3600,
    )
    return {"user": user, "access_token": token}


@app.post("/api/auth/login")
async def login(payload: AuthPayload, response: Response) -> dict[str, Any]:
    user = db.get_user_by_username(payload.username.strip())
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Невірний логін або пароль.")

    token = create_access_token(user["id"], user["username"], user["role"])
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=24 * 3600,
    )
    public_user = {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "created_at": user["created_at"],
    }
    return {"user": public_user, "access_token": token}


@app.post("/api/auth/logout")
async def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie("access_token")
    return {"ok": True}


@app.get("/api/auth/me")
async def me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    active, total = db.count_user_tasks(user["id"])
    return {
        "user": user,
        "quotas": {
            "active_tasks": active,
            "max_active_tasks": MAX_ACTIVE_TASKS_PER_USER,
            "total_tasks": total,
            "max_total_tasks": MAX_TOTAL_TASKS_PER_USER,
        },
        "cluster": lb.get_cluster_capacity(),
    }


# ---------------- Tasks Endpoints (Requirements 1, 2, 3, 5, 6) ----------------


@app.post("/api/tasks", status_code=201)
async def create_task(
    payload: CreateTaskPayload,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    # Requirement 1: Strict backend complexity validation
    if payload.method not in ALLOWED_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Непідтримуваний метод обчислення. Дозволені: {', '.join(ALLOWED_METHODS)}.",
        )
    if payload.matrix_size < MIN_MATRIX_SIZE or payload.matrix_size > MAX_MATRIX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Перевищено обмеження трудомісткості: кількість невідомих n повинна бути в межах від {MIN_MATRIX_SIZE} до {MAX_MATRIX_SIZE}.",
        )
    if payload.timeout_sec < MIN_TIMEOUT_SEC or payload.timeout_sec > MAX_TIMEOUT_SEC:
        raise HTTPException(
            status_code=400,
            detail=f"Ліміт часу обчислення повинен бути в межах від {MIN_TIMEOUT_SEC} до {MAX_TIMEOUT_SEC} секунд.",
        )

    custom_dict: dict[str, Any] | None = None
    if payload.input_mode == "custom":
        if payload.matrix_size > MAX_CUSTOM_MATRIX_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Для ручного введення максимальна розмірність матриці — {MAX_CUSTOM_MATRIX_SIZE}×{MAX_CUSTOM_MATRIX_SIZE}.",
            )
        if not payload.custom_payload:
            raise HTTPException(status_code=400, detail="Потрібно передати матрицю A та вектор b.")
        m = payload.custom_payload.matrix
        v = payload.custom_payload.vector
        if len(m) != payload.matrix_size or len(v) != payload.matrix_size or any(len(r) != payload.matrix_size for r in m):
            raise HTTPException(
                status_code=400,
                detail=f"Розмір матриці A та вектора b має точно відповідати n = {payload.matrix_size}.",
            )
        custom_dict = {"matrix": m, "vector": v}

    # Requirement 3: User active and total task quotas
    active_cnt, total_cnt = db.count_user_tasks(user["id"])
    if active_cnt >= MAX_ACTIVE_TASKS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Досягнуто ліміт одночасних задач ({MAX_ACTIVE_TASKS_PER_USER}). Дочекайтесь завершення або скасуйте активну задачу.",
        )
    if total_cnt >= MAX_TOTAL_TASKS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Досягнуто максимальну кількість збережених задач ({MAX_TOTAL_TASKS_PER_USER}). Видаліть старі записи з історії.",
        )

    # Requirement 6: Global queue limit
    cap = lb.get_cluster_capacity()
    if cap["available_slots"] == 0 and cap["queue_length"] >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=503,
            detail=f"Черга системи переповнена ({MAX_QUEUE_SIZE} задач). Спробуйте пізніше.",
        )

    task_id = uuid.uuid4().hex
    task = db.create_task(
        {
            "id": task_id,
            "user_id": user["id"],
            "username": user["username"],
            "matrix_size": payload.matrix_size,
            "method": payload.method,
            "input_mode": payload.input_mode,
            "seed": payload.seed,
            "custom_payload": custom_dict,
            "timeout_sec": payload.timeout_sec,
            "status": "queued",
            "progress": 0.0,
            "stage": "Ініціалізація задачі...",
            "created_at": time.time(),
        }
    )

    dispatched = await lb.submit_task(task)
    return {"task": dispatched}


@app.get("/api/tasks")
async def list_tasks(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    tasks = db.list_user_tasks(user["id"], limit=MAX_TOTAL_TASKS_PER_USER)
    active, total = db.count_user_tasks(user["id"])
    return {
        "tasks": tasks,
        "quotas": {
            "active_tasks": active,
            "max_active_tasks": MAX_ACTIVE_TASKS_PER_USER,
            "total_tasks": total,
            "max_total_tasks": MAX_TOTAL_TASKS_PER_USER,
        },
        "cluster": lb.get_cluster_capacity(),
    }


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задачу не знайдено.")
    if task["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Немає доступу до цієї задачі.")
    return {"task": task}


@app.get("/api/tasks/{task_id}/stream")
async def stream_task_progress(
    task_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Server-Sent Events (SSE) endpoint for real-time task progress updates (Requirement 2)."""
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задачу не знайдено.")
    if task["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Немає доступу до цієї задачі.")

    async def event_generator():
        q = lb.subscribe(task_id)
        try:
            # Immediately push current state
            current = db.get_task(task_id)
            if current:
                yield f"data: {json.dumps(current, ensure_ascii=False)}\n\n"
                if current["status"] in ("completed", "failed", "cancelled"):
                    return

            while True:
                try:
                    update = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
                    if update["status"] in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # Keep-alive SSE comment
                    yield ": keepalive\n\n"
                    latest = db.get_task(task_id)
                    if not latest or latest["status"] in ("completed", "failed", "cancelled"):
                        if latest:
                            yield f"data: {json.dumps(latest, ensure_ascii=False)}\n\n"
                        break
        finally:
            lb.unsubscribe(task_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    task = db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задачу не знайдено.")
    if task["user_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Немає прав для скасування цієї задачі.")
    updated = await lb.cancel_task(task_id, actor=user["username"])
    return {"task": updated}


@app.delete("/api/tasks/{task_id}")
async def delete_task(
    task_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, bool]:
    deleted = db.delete_task(task_id, user_id=None if user["role"] == "admin" else user["id"])
    if not deleted:
        raise HTTPException(
            status_code=400,
            detail="Задачу не знайдено або вона ще виконується (спочатку скасуйте її).",
        )
    return {"deleted": True}


# ---------------- Admin Panel Endpoints (Requirements 7 & 8) ----------------


@app.get("/api/admin/overview")
async def admin_overview(_: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    return {
        "capacity": lb.get_cluster_capacity(),
        "autoscaler": autoscaler.status_dict(),
        "workers": lb.list_workers(),
        "active_tasks": db.list_active_tasks(),
        "recent_tasks": db.list_all_tasks(limit=40),
        "task_stats": db.get_task_stats(),
        "events": db.list_events(limit=30),
        "server_time": time.time(),
    }


@app.post("/api/admin/scale-up")
async def admin_scale_up(admin: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    res = await autoscaler.scale_up(reason=f"Ручне масштабування адміністратором ({admin['username']})")
    if not res.get("scaled"):
        raise HTTPException(status_code=400, detail=res.get("error", "Помилка масштабування."))
    return res


@app.post("/api/admin/scale-down")
async def admin_scale_down(
    worker_id: str | None = Query(default=None),
    admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    res = await autoscaler.scale_down(
        worker_id=worker_id,
        reason=f"Ручна зупинка адміністратором ({admin['username']})",
    )
    if not res.get("scaled"):
        raise HTTPException(status_code=400, detail=res.get("error", "Помилка масштабування."))
    return res


@app.post("/api/admin/autoscale/toggle")
async def admin_toggle_autoscale(admin: dict[str, Any] = Depends(require_admin)) -> dict[str, Any]:
    autoscaler.enabled = not autoscaler.enabled
    state_str = "увімкнено" if autoscaler.enabled else "вимкнено"
    db.log_event("AUTOSCALE_TOGGLE", admin["username"], f"Автоматичне масштабування {state_str}")
    return {"autoscaler": autoscaler.status_dict()}


@app.post("/api/admin/tasks/{task_id}/cancel")
async def admin_cancel_task(
    task_id: str,
    admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    updated = await lb.cancel_task(task_id, actor=f"admin:{admin['username']}")
    if not updated:
        raise HTTPException(status_code=404, detail="Задачу не знайдено.")
    return {"task": updated}


# ---------------- Internal Worker Callbacks ----------------


@app.post("/internal/tasks/{task_id}/progress")
async def internal_progress(
    task_id: str,
    payload: WorkerProgressPayload,
    x_internal_key: str | None = Header(default=None),
) -> dict[str, bool]:
    _verify_internal_key(x_internal_key)
    await lb.handle_worker_progress(task_id, payload.worker_id, payload.progress, payload.stage)
    return {"ok": True}


@app.post("/internal/tasks/{task_id}/complete")
async def internal_complete(
    task_id: str,
    payload: WorkerCompletePayload,
    x_internal_key: str | None = Header(default=None),
) -> dict[str, bool]:
    _verify_internal_key(x_internal_key)
    await lb.handle_worker_complete(task_id, payload.dict())
    return {"ok": True}
