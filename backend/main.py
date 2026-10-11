"""
二手房价格分析与预测系统 —— Web 服务入口
==========================================
启动：
    python run.py                 # 等价于 uvicorn backend.main:app
    uvicorn backend.main:app --reload --port 8000

三大模块：
  · 数据处理模块   /api/sources  /api/pipeline/*  /api/versions/*  /api/listings
  · 核心算法模块   /api/algorithms  /api/models/*  /api/predict  /api/recommend/*
  · 结果展示模块   /api/analytics/*  /api/experiments/*  /api/docs/*
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import algorithm_router, pipeline_router, query_router
from .settings import WEB_DIR, ensure_dirs

app = FastAPI(
    title="二手房价格分析与预测系统",
    description="基于大数据技术的二手房价格分析与预测模型 —— 数据流水线 / 价格预测 / 房源推荐",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    resp.headers["X-Process-Time-MS"] = str(int((time.time() - t0) * 1000))
    return resp


app.include_router(pipeline_router)
app.include_router(algorithm_router)
app.include_router(query_router)


@app.get("/api/health")
def health() -> dict:
    from .algorithms.trainer import list_models
    from .pipeline.sources import registry
    from .pipeline.store import detail_count, DETAIL_TABLE
    from .pipeline.versioning import versions

    active = versions.active_version()
    return {
        "status": "ok",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": len(registry.list()),
        "versions": len(versions.list()),
        "active_version": (active or {}).get("version_id"),
        "detail_rows": detail_count(DETAIL_TABLE),
        "models": len(list_models()),
    }


@app.on_event("startup")
def _startup() -> None:
    ensure_dirs()


# ---------------------------------------------------------------- 前端静态资源
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    idx = WEB_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "前端文件缺失", "path": str(idx)}, status_code=500)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}", "path": str(request.url.path)},
    )
