"""
模块一 · 数据处理 API（数据源配置 / 导入 / 流水线 / 版本管理 / 数据质量）
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from .. import jobs
from ..pipeline.ingest import detect_encoding, download_url_source
from ..pipeline.runner import DEFAULT_PARAMS, run_pipeline
from ..pipeline.sources import registry
from ..pipeline.store import read_detail
from ..pipeline.versioning import versions

router = APIRouter(prefix="/api", tags=["数据处理"])


# ------------------------------------------------------------ 数据源
@router.get("/sources")
def list_sources() -> dict[str, Any]:
    return {"items": registry.list()}


@router.post("/sources")
def upsert_source(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        src = registry.upsert(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"item": src.to_dict()}


@router.delete("/sources/{source_id}")
def delete_source(source_id: str) -> dict[str, Any]:
    registry.delete(source_id)
    return {"ok": True, "source_id": source_id}


@router.post("/sources/{source_id}/probe")
def probe_source(source_id: str) -> dict[str, Any]:
    """探测数据源：是否存在 / 编码 / 体积（导入前预检）。"""
    try:
        src = registry.get(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from pathlib import Path

    info: dict[str, Any] = {"source_id": source_id, "exists": registry.exists(src)}
    p = Path(registry.resolved_path(src))
    if p.exists():
        info.update({
            "path": str(p),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            "encoding": src.encoding if src.encoding != "auto" else detect_encoding(p),
        })
    elif src.source_type == "url_csv":
        info["downloadable"] = True
    return info


@router.post("/sources/{source_id}/download")
def download_source(source_id: str, force: bool = False) -> dict[str, Any]:
    try:
        src = registry.get(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if src.source_type != "url_csv":
        raise HTTPException(status_code=400, detail="仅 URL 数据源支持下载")
    p = download_url_source(src, force=force)
    return {"path": str(p), "size_mb": round(p.stat().st_size / 1024 / 1024, 2)}


@router.post("/sources/{source_id}/split")
def split_source(source_id: str, chunk_rows: int = 100_000) -> dict[str, Any]:
    """把大文件切分为多个分片文件，体现分批/分区处理。"""
    from pathlib import Path
    import pandas as pd
    from ..settings import RAW_DIR

    try:
        src = registry.get(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    p = Path(registry.resolved_path(src))
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {p}")
    enc = src.encoding if src.encoding != "auto" else detect_encoding(p)
    out_dir = RAW_DIR / f"{p.stem}_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    n, files = 0, []
    reader = pd.read_csv(p, encoding=enc, chunksize=chunk_rows, dtype=str,
                         keep_default_na=False, on_bad_lines="skip")
    for i, ch in enumerate(reader):
        f = out_dir / f"part-{i:04d}.csv"
        ch.to_csv(f, index=False, encoding="utf-8")
        files.append(f.name)
        n += len(ch)
    return {"rows": n, "parts": len(files), "dir": str(out_dir), "files": files[:20]}


# ------------------------------------------------------------ 流水线
@router.get("/pipeline/defaults")
def pipeline_defaults() -> dict[str, Any]:
    from ..pipeline.clean import MISSING_STRATEGIES, OUTLIER_ACTIONS, OUTLIER_STRATEGIES
    return {
        "defaults": DEFAULT_PARAMS,
        "options": {
            "missing_strategy": MISSING_STRATEGIES,
            "outlier_strategy": OUTLIER_STRATEGIES,
            "outlier_action": OUTLIER_ACTIONS,
            "split_protocol": ["time", "random"],
        },
    }


@router.post("/pipeline/run")
def pipeline_run(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    source_id = payload.get("source_id")
    if not source_id:
        raise HTTPException(status_code=400, detail="缺少 source_id")
    params = payload.get("params") or {}
    async_run = payload.get("async", True)

    try:
        registry.get(source_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if async_run:
        job_id = jobs.submit("pipeline", run_pipeline, source_id, params)
        return {"job_id": job_id, "mode": "async"}
    record = run_pipeline(source_id, params)
    return {"mode": "sync", "version": record}


@router.get("/pipeline/status")
def pipeline_status() -> dict[str, Any]:
    from ..pipeline.runner import pipeline_status as ps
    return ps()


# ------------------------------------------------------------ 版本
@router.get("/versions")
def list_versions(limit: int = 50) -> dict[str, Any]:
    return {"items": versions.list(limit)}


@router.get("/versions/{version_id}")
def get_version(version_id: str) -> dict[str, Any]:
    try:
        return {"version": versions.get(version_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/versions/{version_id}/activate")
def activate_version(version_id: str) -> dict[str, Any]:
    try:
        rec = versions.activate(version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"version": rec}


@router.delete("/versions/{version_id}")
def delete_version(version_id: str) -> dict[str, Any]:
    versions.delete(version_id)
    return {"ok": True}


@router.get("/versions/{version_id}/quality")
def data_quality(version_id: str) -> dict[str, Any]:
    """数据质量报告：清洗前后行数、缺失/异常处理明细、特征清单。"""
    try:
        rec = versions.get(version_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "version_id": version_id,
        "stages": rec.get("stages", []),
        "clean_report": rec.get("clean_report", {}),
        "feature_report": rec.get("feature_report", {}),
        "split_meta": rec.get("split_meta", {}),
        "profile_raw": rec.get("data_profile_raw", {}),
        "fingerprint": rec.get("fingerprint", {}),
        "target_stats": rec.get("target_stats", {}),
        "source": {
            "name": rec.get("source_name"), "uri": rec.get("source_uri"),
            "license": rec.get("source_license"), "attribution": rec.get("source_attribution"),
            "raw_checksum": rec.get("raw_checksum"),
        },
        "checksum": rec.get("checksum"),
        "created_at": rec.get("created_at"),
        "seconds_total": rec.get("seconds_total"),
    }


# ------------------------------------------------------------ 明细查询
@router.get("/listings")
def list_listings(
    limit: int = 20,
    offset: int = 0,
    district: str = "",
    min_price: float | None = None,
    max_price: float | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    rooms: int | None = None,
    keyword: str = "",
) -> dict[str, Any]:
    """房源明细条件查询（结果查询模块）。"""
    where: list[str] = []
    if district:
        where.append(f'"district_name" = \'{district}\'')
    if min_price is not None:
        where.append(f'"unit_price" >= {float(min_price)}')
    if max_price is not None:
        where.append(f'"unit_price" <= {float(max_price)}')
    if min_area is not None:
        where.append(f'"area" >= {float(min_area)}')
    if max_area is not None:
        where.append(f'"area" <= {float(max_area)}')
    if rooms:
        where.append(f'"room_total" >= {int(rooms)}')
    if keyword:
        where.append(f'"house_id" LIKE \'%{keyword}%\'')
    clause = " AND ".join(where)

    cols = ["house_id", "district_name", "area", "room_total", "unit_price",
            "total_price", "floor_category", "build_year", "renovation_name",
            "elevator_name", "subway_name", "followers", "days_on_market", "trade_time"]
    try:
        import pandas as pd
        from ..settings import DB_PATH
        conn = __import__("sqlite3").connect(str(DB_PATH))
        cols_sql = ", ".join(f'"{c}"' for c in cols)
        sql = f"SELECT {cols_sql} FROM house_detail"
        if clause:
            sql += f" WHERE {clause}"
        sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        df = pd.read_sql(sql, conn)
        cnt = pd.read_sql(
            "SELECT COUNT(*) AS n FROM house_detail" + (f" WHERE {clause}" if clause else ""), conn
        )["n"].iloc[0]
        conn.close()
        return {"total": int(cnt), "items": df.to_dict(orient="records")}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"查询失败: {exc}") from exc


__all__ = ["router"]
