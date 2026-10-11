"""
模块三 · 结果查询与展示 API（统计指标 / 图表数据 / 任务状态 / 实验记录）
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from .. import jobs
from ..pipeline.store import load_aggregates
from ..pipeline.versioning import versions
from ..settings import EXPERIMENT_DIR

router = APIRouter(prefix="/api", tags=["查询展示"])


# ------------------------------------------------------------ 统计分析
@router.get("/analytics/aggregates")
def aggregates(version_id: str = "") -> dict[str, Any]:
    agg = load_aggregates(version_id or None)
    if agg is None:
        raise HTTPException(status_code=404, detail="尚无汇总数据，请先运行数据流水线")
    return agg


@router.get("/analytics/overview")
def overview(version_id: str = "") -> dict[str, Any]:
    agg = load_aggregates(version_id or None)
    if agg is None:
        return {
            "basic": None,
            "message": "尚无汇总数据：请先在「数据导入」页运行一次数据流水线",
            "active_version": (versions.active_version() or {}).get("version_id"),
        }
    return {
        "basic": agg.get("basic"),
        "generated_at": agg.get("generated_at"),
        "district_top": (agg.get("district") or [])[:13],
        "trend_monthly": agg.get("trend_monthly") or [],
        "area_bucket": agg.get("area_bucket") or [],
        "renovation": agg.get("renovation") or [],
        "elevator": agg.get("elevator") or [],
        "subway": agg.get("subway") or [],
        "building_age_bucket": agg.get("building_age_bucket") or [],
        "active_version": (versions.active_version() or {}).get("version_id"),
    }


@router.get("/analytics/district")
def district(version_id: str = "") -> dict[str, Any]:
    agg = load_aggregates(version_id or None) or {}
    return {"items": agg.get("district") or []}


@router.get("/analytics/trend")
def trend(version_id: str = "", freq: str = "M") -> dict[str, Any]:
    agg = load_aggregates(version_id or None) or {}
    key = "trend_monthly" if freq.upper() == "M" else "trend_quarterly"
    return {"freq": freq, "items": agg.get(key) or []}


@router.get("/analytics/distribution")
def distribution(version_id: str = "", field: str = "unit_price") -> dict[str, Any]:
    agg = load_aggregates(version_id or None) or {}
    key = {"unit_price": "hist_unit_price", "area": "hist_area"}.get(field)
    if not key:
        raise HTTPException(status_code=400, detail=f"不支持的分布字段: {field}")
    return {"field": field, "items": agg.get(key) or []}


@router.get("/analytics/correlation")
def correlation(version_id: str = "") -> dict[str, Any]:
    agg = load_aggregates(version_id or None) or {}
    return agg.get("correlation") or {"columns": [], "matrix": []}


@router.get("/analytics/scatter")
def scatter(version_id: str = "", limit: int = 1200) -> dict[str, Any]:
    agg = load_aggregates(version_id or None) or {}
    return {"items": (agg.get("scatter_area_price") or [])[:limit]}


@router.get("/analytics/geo")
def geo(version_id: str = "", limit: int = 2000) -> dict[str, Any]:
    agg = load_aggregates(version_id or None) or {}
    return {"items": (agg.get("geo_points") or [])[:limit]}


@router.get("/analytics/compare")
def compare_versions(a: str, b: str) -> dict[str, Any]:
    """对比两个数据版本的规模、清洗损失与目标分布（数据版本对比）。"""
    try:
        va, vb = versions.get(a), versions.get(b)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    keys = ["raw_rows", "final_rows", "final_columns", "checksum", "created_at",
            "source_name", "status", "seconds_total"]
    return {
        "a": {"version_id": a, **{k: va.get(k) for k in keys},
              "target_stats": va.get("target_stats"), "params": va.get("params"),
              "stages": [{"stage": s["stage"], "rows_in": s["rows_in"], "rows_out": s["rows_out"],
                          "dropped": s["dropped"], "seconds": s["seconds"]} for s in va.get("stages", [])]},
        "b": {"version_id": b, **{k: vb.get(k) for k in keys},
              "target_stats": vb.get("target_stats"), "params": vb.get("params"),
              "stages": [{"stage": s["stage"], "rows_in": s["rows_in"], "rows_out": s["rows_out"],
                          "dropped": s["dropped"], "seconds": s["seconds"]} for s in vb.get("stages", [])]},
    }


# ------------------------------------------------------------ 任务
@router.get("/jobs")
def jobs_list(limit: int = 20) -> dict[str, Any]:
    return {"items": jobs.list_jobs(limit)}


@router.get("/jobs/{job_id}")
def job_detail(job_id: str) -> dict[str, Any]:
    try:
        return jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ------------------------------------------------------------ 实验记录
@router.get("/experiments")
def experiments() -> dict[str, Any]:
    items = []
    if EXPERIMENT_DIR.exists():
        for p in sorted(EXPERIMENT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            summary = data.get("summary") if isinstance(data, dict) else None
            items.append({
                "file": p.name,
                "name": data.get("experiment") or p.stem,
                "created_at": data.get("created_at"),
                "summary": summary,
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
    return {"items": items}


@router.get("/experiments/{name}")
def experiment_detail(name: str) -> dict[str, Any]:
    p = EXPERIMENT_DIR / (name if name.endswith(".json") else f"{name}.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"实验记录不存在: {name}")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------ 文档
@router.get("/docs/{name}")
def read_doc(name: str) -> dict[str, Any]:
    """读取 docs/ 下的 Markdown 文档，供前端「项目文档」页展示。"""
    from ..settings import BASE_DIR
    safe = name.replace("..", "").replace("/", "").replace("\\", "")
    p = BASE_DIR / "docs" / safe
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"文档不存在: {safe}")
    return {"name": safe, "content": p.read_text(encoding="utf-8")}


@router.get("/docs")
def list_docs() -> dict[str, Any]:
    from ..settings import BASE_DIR
    d = BASE_DIR / "docs"
    items = [{"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
             for f in sorted(d.glob("*.md"))] if d.exists() else []
    return {"items": items}


__all__ = ["router"]
