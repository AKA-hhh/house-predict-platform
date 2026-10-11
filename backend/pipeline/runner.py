"""
流水线编排（Pipeline Runner）
==============================
把「数据源 -> 导入 -> 清洗 -> 特征 -> 划分 -> 存储 -> 汇总」串成**一次可复现运行**，
并登记为一个数据版本（Data Version）。

每一步的输入行数、输出行数、参数、耗时都会写进版本记录；
任一步骤抛错都会被捕获并记入 status='failed' 的版本记录（失败案例留痕）。
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from ..settings import MODEL_DIR, PROCESSED_DIR
from .clean import clean
from .features import build_features
from .ingest import ingest, profile
from .schema import TARGET_COLUMN
from .sources import registry
from .split import split
from .store import (
    DETAIL_TABLE, build_aggregates, detail_count, save_aggregates, write_detail, write_parquet,
)
from .versioning import fingerprint_dataframe, now_str, sha256_file, stage_record, versions

DEFAULT_PARAMS: dict[str, Any] = {
    # 导入
    "chunk_size": 200_000,
    # 清洗
    "missing_strategy": "median",
    "missing_per_column": {},
    "drop_threshold": 0.6,
    "outlier_strategy": "iqr",
    "iqr_k": 1.5,
    "zscore_threshold": 3.0,
    "outlier_action": "winsorize",
    # 划分
    "split_protocol": "time",
    "test_size": 0.2,
    "valid_size": 0.0,
    "seed": 42,
    # 存储
    "store_sqlite": True,
    "store_mode": "replace",
    "save_parquet": True,
}


def run_pipeline(
    source_id: str,
    params: dict[str, Any] | None = None,
    version_id: str | None = None,
) -> dict[str, Any]:
    """执行完整数据流水线，返回数据版本记录。"""
    p = {**DEFAULT_PARAMS, **(params or {})}
    src = registry.get(source_id)
    vid = version_id or versions.new_version_id("ds")

    record: dict[str, Any] = {
        "version_id": vid,
        "created_at": now_str(),
        "source_id": src.source_id,
        "source_name": src.name,
        "source_type": src.source_type,
        "source_uri": src.uri,
        "source_license": src.license,
        "source_attribution": src.attribution,
        "params": p,
        "stages": [],
        "status": "running",
        "active": False,
    }
    t_all = time.time()

    try:
        # ------------------------------------------------ 1. 导入
        t = time.time()
        df, ingest_report = ingest(src, chunk_size=p["chunk_size"])
        record["ingest_report"] = {k: v for k, v in ingest_report.items() if k != "column_map"}
        record["column_map"] = ingest_report.get("column_map", {})
        record["raw_rows"] = int(len(df))
        raw_path = Path(ingest_report.get("file_path", ""))
        if raw_path.exists():
            record["raw_checksum"] = sha256_file(raw_path)
            record["raw_file_size_mb"] = round(raw_path.stat().st_size / 1024 / 1024, 2)
        record["data_profile_raw"] = profile(df)
        record["stages"].append(stage_record(
            "ingest", len(df), len(df), {"chunk_size": p["chunk_size"]},
            time.time() - t, f"编码={ingest_report.get('encoding')} 分块={ingest_report.get('chunks')}",
        ))

        # ------------------------------------------------ 2. 清洗
        t = time.time()
        cleaned, clean_report = clean(
            df,
            missing_strategy=p["missing_strategy"],
            missing_per_column=p.get("missing_per_column") or {},
            drop_threshold=p["drop_threshold"],
            outlier_strategy=p["outlier_strategy"],
            iqr_k=p["iqr_k"],
            zscore_threshold=p["zscore_threshold"],
            outlier_action=p.get("outlier_action", "winsorize"),
        )
        record["clean_report"] = clean_report
        record["stages"].append(stage_record(
            "clean", len(df), len(cleaned), clean_report["params"],
            time.time() - t,
            f"缺失策略={p['missing_strategy']} 异常策略={p['outlier_strategy']}",
        ))

        # ------------------------------------------------ 3. 特征
        t = time.time()
        featured, feat_report = build_features(cleaned)
        record["feature_report"] = feat_report
        record["stages"].append(stage_record(
            "features", len(cleaned), len(featured), {"n_features": feat_report["n_features"]},
            time.time() - t, "防泄漏校验通过",
        ))

        # ------------------------------------------------ 4. 划分
        t = time.time()
        tr, te, va, split_meta = split(
            featured, protocol=p["split_protocol"],
            test_size=p["test_size"], valid_size=p["valid_size"], seed=p["seed"],
        )
        record["split_meta"] = split_meta
        record["stages"].append(stage_record(
            "split", len(featured), len(tr) + len(te) + (len(va) if va is not None else 0),
            {k: split_meta[k] for k in ("protocol", "test_size", "valid_size", "seed") if k in split_meta},
            time.time() - t, f"train={len(tr)} test={len(te)}",
        ))

        # ------------------------------------------------ 5. 存储
        t = time.time()
        pq_path = ""
        if p.get("save_parquet", True):
            pq_path = write_parquet(featured, PROCESSED_DIR / f"{vid}.parquet")
            record["parquet_path"] = pq_path

        sqlite_info = {}
        if p.get("store_sqlite", True):
            sqlite_info = write_detail(featured, DETAIL_TABLE, mode=p["store_mode"],
                                       chunk_size=50_000)
            record["sqlite"] = sqlite_info
            registry.mark_run(src.source_id, len(featured))

        # 划分数据集另存（供训练快速加载）
        split_dir = MODEL_DIR / vid
        split_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(tr, split_dir / "train.parquet")
        write_parquet(te, split_dir / "test.parquet")
        if va is not None:
            write_parquet(va, split_dir / "valid.parquet")
        record["split_dir"] = str(split_dir)

        record["stages"].append(stage_record(
            "store", len(featured), len(featured),
            {"parquet": bool(pq_path), "sqlite": p.get("store_sqlite"), "mode": p.get("store_mode")},
            time.time() - t, f"sqlite_rows={sqlite_info.get('table_rows', 0)}",
        ))

        # ------------------------------------------------ 6. 汇总（批处理）
        t = time.time()
        agg = build_aggregates(featured)
        agg_paths = save_aggregates(agg, vid)
        record["aggregate_paths"] = agg_paths
        record["stages"].append(stage_record(
            "aggregate", len(featured), len(featured),
            {"tables": 11}, time.time() - t, "预聚合指标已落盘",
        ))

        # ------------------------------------------------ 7. 指纹
        record["final_rows"] = int(len(featured))
        record["final_columns"] = int(featured.shape[1])
        record["fingerprint"] = fingerprint_dataframe(featured)
        record["checksum"] = sha256_file(pq_path) if pq_path else ""
        record["target_stats"] = {
            "column": TARGET_COLUMN,
            "mean": round(float(featured[TARGET_COLUMN].mean()), 2),
            "median": round(float(featured[TARGET_COLUMN].median()), 2),
            "std": round(float(featured[TARGET_COLUMN].std()), 2),
            "min": round(float(featured[TARGET_COLUMN].min()), 2),
            "max": round(float(featured[TARGET_COLUMN].max()), 2),
        }
        record["status"] = "success"
        record["seconds_total"] = round(time.time() - t_all, 2)
        versions.save(record)

        # 最新一次成功的运行自动成为"当前激活版本"
        versions.activate(vid)
        return record

    except Exception as exc:  # noqa: BLE001 —— 失败必须留痕，便于「失败案例」章节
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-4000:]
        record["seconds_total"] = round(time.time() - t_all, 2)
        versions.save(record)
        raise


def pipeline_status() -> dict[str, Any]:
    """当前系统数据状态摘要（供前端首屏展示）。"""
    active = versions.active_version()
    return {
        "active_version": (active or {}).get("version_id"),
        "active_rows": (active or {}).get("final_rows"),
        "active_source": (active or {}).get("source_name"),
        "active_created_at": (active or {}).get("created_at"),
        "versions_total": len(versions.list()),
        "sqlite_rows": detail_count(DETAIL_TABLE),
    }


__all__ = ["run_pipeline", "pipeline_status", "DEFAULT_PARAMS"]
