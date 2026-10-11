"""后台任务管理：把耗时的流水线/训练/评估放到线程池执行，前端轮询进度。"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
MAX_JOBS = 60


def submit(kind: str, fn: Callable[..., Any], *args, **kwargs) -> str:
    job_id = f"{kind}-{uuid.uuid4().hex[:8]}"
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "finished_at": None,
            "seconds": None,
            "result": None,
            "error": None,
            "logs": [],
        }
        if len(_jobs) > MAX_JOBS:
            for k in list(_jobs.keys())[:-MAX_JOBS]:
                if _jobs[k]["status"] in ("success", "failed"):
                    _jobs.pop(k, None)

    def _run() -> None:
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()
        try:
            res = fn(*args, **kwargs)
            with _lock:
                _jobs[job_id].update({
                    "status": "success",
                    "result": _shrink(res),
                    "seconds": round(time.time() - t0, 2),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _jobs[job_id].update({
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-3000:],
                    "seconds": round(time.time() - t0, 2),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

    _executor.submit(_run)
    return job_id


def _shrink(obj: Any, depth: int = 0) -> Any:
    """裁剪超大返回值，避免把整份数据塞进任务结果。"""
    if depth > 3:
        return "..."
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("traceback",):
                continue
            if isinstance(v, list) and len(v) > 60:
                out[k] = [f"<{len(v)} items truncated>"] + [_shrink(x, depth + 1) for x in v[:3]]
            else:
                out[k] = _shrink(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_shrink(x, depth + 1) for x in obj[:60]]
    return obj


def get(job_id: str) -> dict[str, Any]:
    with _lock:
        if job_id not in _jobs:
            raise KeyError(f"任务不存在: {job_id}")
        return dict(_jobs[job_id])


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with _lock:
        items = list(_jobs.values())[::-1][:limit]
        return [{k: v for k, v in j.items() if k not in ("result", "traceback")} for j in items]


__all__ = ["submit", "get", "list_jobs"]
