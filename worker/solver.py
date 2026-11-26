"""CPU-intensive SLAE (Systems of Linear Algebraic Equations) solver with progress, cancellation & timeout."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np


class TaskCancelledError(Exception):
    """Raised when a user or admin cancels a running task."""


class TaskTimeoutError(Exception):
    """Raised when task execution exceeds the configured time limit."""


class SingularMatrixError(Exception):
    """Raised when the linear system matrix is singular or numerically unstable."""


ProgressCallback = Callable[[float, str], None]


def _build_system(
    matrix_size: int,
    input_mode: str,
    seed: int | None,
    custom_payload: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct matrix A (n x n) and right-hand side vector b (n)."""
    if input_mode == "custom" and custom_payload:
        A = np.array(custom_payload["matrix"], dtype=np.float64)
        b = np.array(custom_payload["vector"], dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
            raise ValueError("Некоректні розміри матриці A або вектора b.")
        return A, b

    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % (2**31))
    # Generate a well-conditioned strictly diagonally dominant matrix so both Gauss and Jacobi converge reliably.
    A = rng.uniform(-1.0, 1.0, size=(matrix_size, matrix_size))
    row_sums = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, row_sums + rng.uniform(1.0, 2.5, size=matrix_size))
    # True solution x_true in [-10, 10] -> b = A @ x_true
    x_true = rng.uniform(-10.0, 10.0, size=matrix_size)
    b = A @ x_true
    return A, b


def _solve_gauss_partial_pivoting(
    A_orig: np.ndarray,
    b_orig: np.ndarray,
    timeout_sec: float,
    start_time: float,
    cancel_event: threading.Event,
    on_progress: ProgressCallback,
) -> tuple[np.ndarray, int]:
    """Solve Ax = b via Gaussian elimination with partial pivoting (O(n^3))."""
    n = A_orig.shape[0]
    A = A_orig.copy()
    b = b_orig.copy()

    report_step = max(1, n // 40)
    last_report = time.monotonic()

    # Forward elimination: 10% -> 85%
    for k in range(n - 1):
        if cancel_event.is_set():
            raise TaskCancelledError("Задачу скасовано користувачем.")
        if (time.monotonic() - start_time) > timeout_sec:
            raise TaskTimeoutError(f"Перевищено ліміт часу обчислення ({timeout_sec:.0f} с).")

        # Partial pivoting
        pivot_rel = int(np.argmax(np.abs(A[k:, k])))
        pivot_row = k + pivot_rel
        if abs(A[pivot_row, k]) < 1e-14:
            raise SingularMatrixError(f"Матриця вироджена (нульовий головний елемент у стовпці {k + 1}).")

        if pivot_row != k:
            A[[k, pivot_row], k:] = A[[pivot_row, k], k:]
            b[k], b[pivot_row] = b[pivot_row], b[k]

        # Vectorized elimination of submatrix below pivot
        factors = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k] = 0.0
        A[k + 1 :, k + 1 :] -= np.outer(factors, A[k, k + 1 :])
        b[k + 1 :] -= factors * b[k]

        if k % report_step == 0 or (time.monotonic() - last_report) >= 0.35:
            pct = 10.0 + (75.0 * (k + 1) / max(1, n - 1))
            on_progress(round(pct, 1), f"Прямий хід методу Гауса ({k + 1}/{n})")
            last_report = time.monotonic()

    if abs(A[n - 1, n - 1]) < 1e-14:
        raise SingularMatrixError("Матриця вироджена (нульовий останній діагональний елемент).")

    # Back substitution: 85% -> 95%
    on_progress(86.0, "Зворотний хід методу Гауса")
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if cancel_event.is_set():
            raise TaskCancelledError("Задачу скасовано користувачем.")
        if (time.monotonic() - start_time) > timeout_sec:
            raise TaskTimeoutError(f"Перевищено ліміт часу обчислення ({timeout_sec:.0f} с).")
        s = float(np.dot(A[i, i + 1 :], x[i + 1 :]))
        x[i] = (b[i] - s) / A[i, i]

    on_progress(95.0, "Обчислення нев'язки та перевірка точності")
    return x, n


def _solve_jacobi(
    A: np.ndarray,
    b: np.ndarray,
    timeout_sec: float,
    start_time: float,
    cancel_event: threading.Event,
    on_progress: ProgressCallback,
    max_iterations: int = 800,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int]:
    """Solve Ax = b via Jacobi iterative method (O(k * n^2))."""
    n = A.shape[0]
    diag = np.diag(A)
    if np.any(np.abs(diag) < 1e-14):
        raise SingularMatrixError("Нульовий елемент на головній діагоналі — метод Якобі неможливий.")

    R = A.copy()
    np.fill_diagonal(R, 0.0)
    x = np.zeros(n, dtype=np.float64)
    last_report = time.monotonic()

    iterations = 0
    for it in range(1, max_iterations + 1):
        if cancel_event.is_set():
            raise TaskCancelledError("Задачу скасовано користувачем.")
        if (time.monotonic() - start_time) > timeout_sec:
            raise TaskTimeoutError(f"Перевищено ліміт часу обчислення ({timeout_sec:.0f} с).")

        x_new = (b - R @ x) / diag
        diff = float(np.linalg.norm(x_new - x, ord=np.inf))
        x = x_new
        iterations = it

        if diff < tol:
            on_progress(95.0, f"Збіжність досягнуто на ітерації {it}")
            break

        if it % 10 == 0 or (time.monotonic() - last_report) >= 0.35:
            pct = min(94.0, 10.0 + 84.0 * (it / max_iterations))
            on_progress(round(pct, 1), f"Ітерації методу Якобі ({it}/{max_iterations}, Δ={diff:.2e})")
            last_report = time.monotonic()

    on_progress(96.0, "Обчислення нев'язки та перевірка точності")
    return x, iterations


def solve_slae(
    matrix_size: int,
    method: str,
    input_mode: str,
    seed: int | None,
    custom_payload: dict[str, Any] | None,
    timeout_sec: float,
    cancel_event: threading.Event,
    on_progress: ProgressCallback,
) -> dict[str, Any]:
    """Full SLAE computation pipeline with progress updates, cancellation, and residual verification."""
    start_time = time.monotonic()
    on_progress(2.0, f"Формування системи рівнянь ({matrix_size}×{matrix_size})")

    if cancel_event.is_set():
        raise TaskCancelledError("Задачу скасовано до початку обчислення.")

    A, b = _build_system(matrix_size, input_mode, seed, custom_payload)
    on_progress(10.0, "Матрицю підготовлено, запуск обчислення")

    if method == "jacobi":
        x, iterations = _solve_jacobi(A, b, timeout_sec, start_time, cancel_event, on_progress)
    else:
        x, iterations = _solve_gauss_partial_pivoting(A, b, timeout_sec, start_time, cancel_event, on_progress)

    # Residual Euclidean norm ||Ax - b||_2
    residual_norm = float(np.linalg.norm(A @ x - b, ord=2))
    elapsed_ms = round((time.monotonic() - start_time) * 1000.0, 2)

    preview_count = min(12, len(x))
    solution_preview = [round(float(v), 6) for v in x[:preview_count]]

    on_progress(100.0, "Обчислення успішно завершено")
    return {
        "execution_time_ms": elapsed_ms,
        "residual_norm": residual_norm,
        "iterations": iterations,
        "solution_preview": solution_preview,
    }
