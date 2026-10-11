#!/usr/bin/env python
"""
独立数据导入 / 更新工具（命令行）
==================================
即使不启动 Web 服务，也可以用本工具完成数据的导入、更新与版本管理，
适合放进定时任务（cron / Windows 计划任务）做周期批处理。

子命令：
  sources                      列出全部数据源
  probe    --source ID         探测数据源（是否存在 / 编码 / 体积）
  download --source ID         下载 URL 数据源到本地 raw 目录
  split    --source ID         把大文件切分为分片，便于分批处理
  run      --source ID         运行完整数据流水线，产出新数据版本
  update   --source ID --csv F 增量更新：把新批次数据合并进明细表
  versions                     列出全部数据版本
  activate --version ID        激活某个数据版本
  export   --version ID --out F 导出某版本为 CSV / Parquet

示例：
  python tools/etl.py run --source lianjia_bj_raw --missing median --outlier iqr --split time
  python tools/etl.py update --source lianjia_bj_raw --csv data/raw/new_batch.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import pandas as pd  # noqa: E402

from backend.pipeline.clean import clean  # noqa: E402
from backend.pipeline.features import build_features  # noqa: E402
from backend.pipeline.ingest import detect_encoding, download_url_source, ingest, normalize_columns, coerce_numeric  # noqa: E402
from backend.pipeline.runner import run_pipeline  # noqa: E402
from backend.pipeline.sources import registry  # noqa: E402
from backend.pipeline.store import DETAIL_TABLE, write_detail  # noqa: E402
from backend.pipeline.versioning import versions  # noqa: E402
from backend.settings import RAW_DIR  # noqa: E402


def cmd_sources(_args) -> None:
    for s in registry.list():
        flag = "就绪" if s["exists"] else "待下载"
        print(f"  {s['source_id']:<24} {s['source_type']:<10} [{flag}] {s['name']}")
        print(f"      uri: {s['uri']}")


def cmd_probe(args) -> None:
    src = registry.get(args.source)
    p = Path(registry.resolved_path(src))
    info = {"source_id": args.source, "exists": p.exists()}
    if p.exists():
        info["size_mb"] = round(p.stat().st_size / 1024 / 1024, 2)
        info["encoding"] = src.encoding if src.encoding != "auto" else detect_encoding(p)
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_download(args) -> None:
    src = registry.get(args.source)
    p = download_url_source(src, force=args.force)
    print(f"已下载到 {p}  ({p.stat().st_size / 1024 / 1024:.2f} MB)")


def cmd_split(args) -> None:
    src = registry.get(args.source)
    p = Path(registry.resolved_path(src))
    enc = src.encoding if src.encoding != "auto" else detect_encoding(p)
    out_dir = RAW_DIR / f"{p.stem}_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, ch in enumerate(pd.read_csv(p, encoding=enc, chunksize=args.rows, dtype=str,
                                       keep_default_na=False, on_bad_lines="skip")):
        ch.to_csv(out_dir / f"part-{i:04d}.csv", index=False, encoding="utf-8")
        n += len(ch)
        print(f"  part-{i:04d}.csv  {len(ch)} 行")
    print(f"共 {n} 行，输出目录 {out_dir}")


def cmd_run(args) -> None:
    params = {
        "missing_strategy": args.missing,
        "outlier_strategy": args.outlier,
        "outlier_action": args.outlier_action,
        "split_protocol": args.split,
        "test_size": args.test_size,
        "iqr_k": args.iqr_k,
        "drop_threshold": args.drop_threshold,
        "store_mode": args.store_mode,
    }
    rec = run_pipeline(args.source, params)
    print("\n=== 数据版本 ===")
    print(f"  版本号     : {rec['version_id']}")
    print(f"  原始行数   : {rec['raw_rows']:,}")
    print(f"  最终行数   : {rec['final_rows']:,}")
    print(f"  最终列数   : {rec['final_columns']}")
    print(f"  总耗时     : {rec['seconds_total']}s")
    print(f"  校验和     : {rec.get('checksum', '')[:32]}")
    print("  阶段明细:")
    for s in rec["stages"]:
        print(f"    {s['stage']:<10} {s['rows_in']:>9,} -> {s['rows_out']:>9,}  "
              f"剔除 {s['dropped']:>7,}  {s['seconds']:>7.2f}s  {s['notes']}")


def cmd_update(args) -> None:
    """增量更新：读取新批次 CSV，执行同样的清洗与特征工程后 upsert 进明细表。"""
    src = registry.get(args.source)
    p = Path(args.csv)
    if not p.exists():
        raise SystemExit(f"文件不存在: {p}")
    enc = detect_encoding(p)
    print(f"读取 {p} (编码 {enc}) …")
    chunks = []
    for ch in pd.read_csv(p, encoding=enc, chunksize=200_000, dtype=str,
                          keep_default_na=False, na_values=["", "NA", "null"], on_bad_lines="skip"):
        ch, _ = normalize_columns(ch)
        chunks.append(coerce_numeric(ch))
    df = pd.concat(chunks, ignore_index=True)
    print(f"  {len(df):,} 行")

    cleaned, rep = clean(df, missing_strategy=args.missing, outlier_strategy=args.outlier)
    featured, freq = build_features(cleaned)
    info = write_detail(featured, DETAIL_TABLE, mode="upsert", chunk_size=50_000)
    print(f"  清洗后 {len(featured):,} 行，upsert 完成，明细表累计 {info['table_rows']:,} 行")
    print(json.dumps({"clean": rep["rows_dropped_total"], "features": freq["n_features"]}, ensure_ascii=False))


def cmd_versions(_args) -> None:
    for v in versions.list():
        mark = "★当前" if v.get("active") else "   "
        print(f"  {mark} {v['version_id']:<32} {v.get('created_at', '')}  "
              f"{v.get('raw_rows', 0):>9,} -> {v.get('final_rows', 0):>9,}  {v.get('status')}")


def cmd_activate(args) -> None:
    rec = versions.activate(args.version)
    print(f"已激活 {rec['version_id']}")


def cmd_export(args) -> None:
    rec = versions.get(args.version)
    pq = Path(rec["parquet_path"])
    if not pq.exists():
        raise SystemExit(f"Parquet 快照缺失: {pq}")
    out = Path(args.out)
    df = pd.read_parquet(pq)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已导出 {len(df):,} 行 -> {out}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="二手房数据导入 / 更新工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sources").set_defaults(func=cmd_sources)

    sp = sub.add_parser("probe"); sp.add_argument("--source", required=True); sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("download")
    sp.add_argument("--source", required=True); sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("split")
    sp.add_argument("--source", required=True); sp.add_argument("--rows", type=int, default=100_000)
    sp.set_defaults(func=cmd_split)

    sp = sub.add_parser("run")
    sp.add_argument("--source", required=True)
    sp.add_argument("--missing", default="median", choices=["drop", "mean", "median", "mode", "constant", "ffill"])
    sp.add_argument("--outlier", default="iqr", choices=["iqr", "zscore", "range", "none"])
    sp.add_argument("--outlier-action", default="winsorize", dest="outlier_action",
                    choices=["winsorize", "clip_median", "drop"])
    sp.add_argument("--split", default="time", choices=["time", "random"])
    sp.add_argument("--test-size", type=float, default=0.2, dest="test_size")
    sp.add_argument("--iqr-k", type=float, default=1.5, dest="iqr_k")
    sp.add_argument("--drop-threshold", type=float, default=0.6, dest="drop_threshold")
    sp.add_argument("--store-mode", default="replace", dest="store_mode", choices=["replace", "upsert"])
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("update")
    sp.add_argument("--source", required=True); sp.add_argument("--csv", required=True)
    sp.add_argument("--missing", default="median"); sp.add_argument("--outlier", default="iqr")
    sp.set_defaults(func=cmd_update)

    sub.add_parser("versions").set_defaults(func=cmd_versions)

    sp = sub.add_parser("activate"); sp.add_argument("--version", required=True); sp.set_defaults(func=cmd_activate)

    sp = sub.add_parser("export")
    sp.add_argument("--version", required=True); sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_export)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
