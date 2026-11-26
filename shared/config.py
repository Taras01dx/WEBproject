"""Centralized configuration and domain limits for the SLAE Compute Cluster."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "cluster.db")))

# Security & Auth
JWT_SECRET = os.getenv("JWT_SECRET", "slae-cluster-dev-secret-change-in-production-2026")
JWT_EXPIRES_HOURS = int(os.getenv("JWT_EXPIRES_HOURS", "24"))
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "internal-cluster-token-2026")

# Requirement 1: Complexity limits per task
MIN_MATRIX_SIZE = int(os.getenv("MIN_MATRIX_SIZE", "2"))
MAX_MATRIX_SIZE = int(os.getenv("MAX_MATRIX_SIZE", "1500"))
MAX_CUSTOM_MATRIX_SIZE = int(os.getenv("MAX_CUSTOM_MATRIX_SIZE", "10"))
MIN_TIMEOUT_SEC = int(os.getenv("MIN_TIMEOUT_SEC", "5"))
MAX_TIMEOUT_SEC = int(os.getenv("MAX_TIMEOUT_SEC", "45"))
ALLOWED_METHODS = ("gauss", "jacobi")

# Requirement 3: User task limits
MAX_ACTIVE_TASKS_PER_USER = int(os.getenv("MAX_ACTIVE_TASKS_PER_USER", "4"))
MAX_TOTAL_TASKS_PER_USER = int(os.getenv("MAX_TOTAL_TASKS_PER_USER", "50"))

# Requirement 5 & 6: Worker capacity and queue
WORKER_MAX_CONCURRENCY = int(os.getenv("WORKER_MAX_CONCURRENCY", "2"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "30"))
HEARTBEAT_INTERVAL_SEC = float(os.getenv("HEARTBEAT_INTERVAL_SEC", "3.0"))
WORKER_TIMEOUT_SEC = float(os.getenv("WORKER_TIMEOUT_SEC", "12.0"))

# Requirement 7: Horizontal Autoscaling limits
AUTOSCALE_ENABLED = os.getenv("AUTOSCALE_ENABLED", "true").lower() in ("1", "true", "yes")
AUTOSCALE_MODE = os.getenv("AUTOSCALE_MODE", "local")  # "local" (subprocess) or "docker"
MIN_WORKERS = int(os.getenv("MIN_WORKERS", "2"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
AUTOSCALE_CHECK_INTERVAL_SEC = float(os.getenv("AUTOSCALE_CHECK_INTERVAL_SEC", "3.0"))
SCALE_UP_UTILIZATION_THRESHOLD = float(os.getenv("SCALE_UP_UTILIZATION_THRESHOLD", "0.80"))
SCALE_UP_COOLDOWN_SEC = float(os.getenv("SCALE_UP_COOLDOWN_SEC", "8.0"))
SCALE_DOWN_IDLE_SEC = float(os.getenv("SCALE_DOWN_IDLE_SEC", "45.0"))
DYNAMIC_WORKER_PORT_START = int(os.getenv("DYNAMIC_WORKER_PORT_START", "8003"))

# Docker autoscaling settings (used when AUTOSCALE_MODE == "docker")
DOCKER_SOCKET_PATH = os.getenv("DOCKER_SOCKET_PATH", "/var/run/docker.sock")
DOCKER_WORKER_IMAGE = os.getenv("DOCKER_WORKER_IMAGE", "slae-compute-cluster-worker:latest")
DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "slae_cluster_net")
BALANCER_INTERNAL_URL = os.getenv("BALANCER_INTERNAL_URL", "http://127.0.0.1:8000")

# Initial static workers (comma-separated "id=url" pairs)
DEFAULT_INITIAL_WORKERS = os.getenv(
    "INITIAL_WORKERS",
    "worker-1=http://127.0.0.1:8001,worker-2=http://127.0.0.1:8002",
)


def parse_initial_workers(raw: str = DEFAULT_INITIAL_WORKERS) -> list[tuple[str, str]]:
    """Parse comma-separated worker definitions: 'worker-1=http://...,worker-2=http://...'."""
    result: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        wid, url = item.split("=", 1)
        result.append((wid.strip(), url.strip().rstrip("/")))
    return result
