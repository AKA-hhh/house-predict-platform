"""
Stage 5 · 数据存储与批处理汇总（Store）
=========================================
两级存储：

  1. **明细层**：清洗后全量数据
       - Parquet 列存快照（按数据版本落盘，供复现与版本对比）
       - SQLite `house_detail` 表（支持 SQL 查询、增量 upsert）
  2. **汇总层**：面向展示的预聚合指标（批处理产物，避免前端每次全表扫描）
       - 区域统计、成交趋势、面积/楼龄分布、价格直方图、相关系数矩阵
       - 同时写入 SQLite `agg_*` 表与 JSON 报告

「增量更新」：以 house_id 为主键先删后插（upsert），支持追加新批次成交数据。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..settings import DB_PATH, PROCESSED_DIR, REPORT_DIR
from .schema import TARGET_COLUMN

DETAIL_TABLE = "house_detail"


# ------------------------------------------------------------ 连接管理
def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ------------------------------------------------------------ 明细写入
def write_parquet(df: pd.DataFrame, path: Path | str) -> str:
    """写入 Parquet 列存快照（时间列统一为字符串，避免时区/类型歧义）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    out.to_parquet(path, index=False, compression="snappy")
    return str(path)


def read_parquet(path: Path | str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def write_detail(
    df: pd.DataFrame,
    table: str = DETAIL_TABLE,
    mode: str = "replace",
    chunk_size: int = 50_000,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """把明细写入 SQLite。

    mode='replace' 全量重建；mode='upsert' 按 house_id 增量合并。
    """
    t0 = time.time()
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
        elif out[c].dtype == "category":
            out[c] = out[c].astype(str)

    conn = connect(db_path)
    try:
        if mode == "upsert" and "house_id" in out.columns:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (house_id TEXT PRIMARY KEY)")
            existing = pd.read_sql(f"SELECT house_id FROM {table}", conn)
            cur_ids = set(existing["house_id"].astype(str)) if len(existing) else set()
            new_ids = set(out["house_id"].astype(str))
            if cur_ids & new_ids:
                conn.executemany(
                    f"DELETE FROM {table} WHERE house_id = ?",
                    [(i,) for i in (cur_ids & new_ids)],
                )
                # 重建为无主键表以便 pandas 追加
                conn.execute(f"ALTER TABLE {table} RENAME TO {table}_tmp")
                conn.execute(f"DROP TABLE {table}_tmp")
            conn.commit()
            out.to_sql(table, conn, if_exists="append", index=False, chunksize=chunk_size)
            action = "upsert"
        else:
            out.to_sql(table, conn, if_exists="replace", index=False, chunksize=chunk_size)
            action = "replace"

        # 常用查询索引
        for col in ("district_name", "trade_time", "unit_price", "area", "house_id"):
            if col in out.columns:
                conn.execute(
                    f'CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}("{col}")'
                )
        conn.commit()
        total = int(pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", conn)["n"].iloc[0])
    finally:
        conn.close()

    return {
        "table": table,
        "mode": action,
        "written_rows": int(len(out)),
        "table_rows": total,
        "seconds": round(time.time() - t0, 3),
    }


def read_detail(
    table: str = DETAIL_TABLE,
    limit: int | None = None,
    where: str = "",
    columns: list[str] | None = None,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    sql = f"SELECT {cols} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    conn = connect(db_path)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def detail_count(table: str = DETAIL_TABLE, db_path: Path | str | None = None) -> int:
    conn = connect(db_path)
    try:
        try:
            return int(pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", conn)["n"].iloc[0])
        except Exception:
            return 0
    finally:
        conn.close()


# ============================================================ 批处理汇总
def _agg_basic(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "avg_unit_price": round(float(df[TARGET_COLUMN].mean()), 2),
        "median_unit_price": round(float(df[TARGET_COLUMN].median()), 2),
        "p10_unit_price": round(float(df[TARGET_COLUMN].quantile(0.1)), 2),
        "p90_unit_price": round(float(df[TARGET_COLUMN].quantile(0.9)), 2),
        "avg_total_price": round(float(df["total_price"].mean()), 2),
        "avg_area": round(float(df["area"].mean()), 2),
        "median_area": round(float(df["area"].median()), 2),
        "avg_building_age": round(float(df["building_age"].mean()), 2) if "building_age" in df else None,
        "avg_days_on_market": round(float(df["days_on_market"].mean()), 2) if "days_on_market" in df else None,
        "avg_followers": round(float(df["followers"].mean()), 2) if "followers" in df else None,
        "n_districts": int(df["district_name"].nunique()) if "district_name" in df else 0,
        "n_communities": int(df["community_id"].nunique()) if "community_id" in df else 0,
        "trade_period": [
            str(pd.to_datetime(df["trade_time"]).min())[:10],
            str(pd.to_datetime(df["trade_time"]).max())[:10],
        ] if "trade_time" in df else None,
    }


def _agg_district(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "district_name" not in df.columns:
        return []
    g = (
        df.groupby("district_name")
        .agg(
            count=("house_id", "count"),
            avg_unit_price=(TARGET_COLUMN, "mean"),
            median_unit_price=(TARGET_COLUMN, "median"),
            avg_total_price=("total_price", "mean"),
            avg_area=("area", "mean"),
            avg_building_age=("building_age", "mean") if "building_age" in df else ("area", "mean"),
            avg_days_on_market=("days_on_market", "mean") if "days_on_market" in df else ("area", "mean"),
            community_avg_price=("community_avg_price", "mean") if "community_avg_price" in df else ("area", "mean"),
        )
        .reset_index()
    )
    g = g.sort_values("avg_unit_price", ascending=False)
    for c in g.columns:
        if c != "district_name" and c != "count":
            g[c] = g[c].round(2)
    return g.to_dict(orient="records")


def _agg_trend(df: pd.DataFrame, freq: str = "M") -> list[dict[str, Any]]:
    if "trade_time" not in df.columns:
        return []
    d = df.copy()
    d["trade_time"] = pd.to_datetime(d["trade_time"], errors="coerce")
    d = d.dropna(subset=["trade_time"])
    g = (
        d.set_index("trade_time")
        .resample("ME" if freq == "M" else "QE")
        .agg(count=("house_id", "count"),
             avg_unit_price=(TARGET_COLUMN, "mean"),
             median_unit_price=(TARGET_COLUMN, "median"),
             avg_total_price=("total_price", "mean"),
             avg_area=("area", "mean"))
        .reset_index()
    )
    g["period"] = g["trade_time"].dt.strftime("%Y-%m")
    g = g[g["count"] > 0]
    for c in ("avg_unit_price", "median_unit_price", "avg_total_price", "avg_area"):
        g[c] = g[c].round(2)
    return g[["period", "count", "avg_unit_price", "median_unit_price", "avg_total_price", "avg_area"]].to_dict(orient="records")


def _agg_hist(df: pd.DataFrame, col: str = TARGET_COLUMN, bins: int = 40) -> list[dict[str, Any]]:
    if col not in df.columns:
        return []
    s = df[col].dropna()
    s = s[s.between(s.quantile(0.001), s.quantile(0.999))]
    cnt, edges = np.histogram(s, bins=bins)
    return [
        {"bin_start": round(float(edges[i]), 1),
         "bin_end": round(float(edges[i + 1]), 1),
         "label": f"{int(edges[i] / 1000)}k-{int(edges[i + 1] / 1000)}k",
         "count": int(cnt[i])}
        for i in range(len(cnt))
    ]


def _agg_group(df: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    if col not in df.columns:
        return []
    g = (df.groupby(col)
           .agg(count=("house_id", "count"), avg_unit_price=(TARGET_COLUMN, "mean"),
                median_unit_price=(TARGET_COLUMN, "median"))
           .reset_index().sort_values(col))
    g["avg_unit_price"] = g["avg_unit_price"].round(2)
    g["median_unit_price"] = g["median_unit_price"].round(2)
    return g.to_dict(orient="records")


def _agg_group_by(df: pd.DataFrame, col: str, order: list[str]) -> list[dict[str, Any]]:
    recs = _agg_group(df, col)
    if order:
        rank = {k: i for i, k in enumerate(order)}
        recs.sort(key=lambda r: rank.get(r[col], 999))
    return recs


def _agg_corr(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    use = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(use) < 2:
        return {"columns": [], "matrix": []}
    c = df[use].corr().round(3)
    return {"columns": list(c.columns), "matrix": c.values.tolist()}


def _agg_scatter(df: pd.DataFrame, n: int = 1200, seed: int = 7) -> list[dict[str, Any]]:
    if not {"area", TARGET_COLUMN}.issubset(df.columns):
        return []
    s = df.sample(min(n, len(df)), random_state=seed)
    return [
        {"area": round(float(a), 2), "unit_price": round(float(p), 1),
         "district": str(d), "total_price": round(float(t), 1)}
        for a, p, d, t in zip(s["area"], s[TARGET_COLUMN],
                              s.get("district_name", pd.Series(["未知"] * len(s))),
                              s.get("total_price", pd.Series([0] * len(s))))
    ]


def _agg_geo(df: pd.DataFrame, n: int = 2000, seed: int = 11) -> list[dict[str, Any]]:
    if not {"lng", "lat", TARGET_COLUMN}.issubset(df.columns):
        return []
    s = df.dropna(subset=["lng", "lat"]).sample(min(n, len(df)), random_state=seed)
    return [
        {"lng": round(float(x), 5), "lat": round(float(y), 5), "unit_price": round(float(p), 0)}
        for x, y, p in zip(s["lng"], s["lat"], s[TARGET_COLUMN])
    ]


def build_aggregates(df: pd.DataFrame, top_n_corr: int = 16) -> dict[str, Any]:
    """计算全部展示用预聚合指标（批处理）。"""
    t0 = time.time()
    cols = ["area", "total_price", TARGET_COLUMN, "living_room", "drawing_room",
            "bath_room", "kitchen", "building_age", "days_on_market", "followers",
            "community_avg_price", "ladder_ratio", "dist_to_center"]
    area_order = ["<50", "50-70", "70-90", "90-120", "120-150", "150-200", "200+"]
    agg: dict[str, Any] = {
        "basic": _agg_basic(df),
        "district": _agg_district(df),
        "trend_monthly": _agg_trend(df, "M"),
        "trend_quarterly": _agg_trend(df, "Q"),
        "hist_unit_price": _agg_hist(df, TARGET_COLUMN, 40),
        "hist_area": _agg_hist(df, "area", 30) if "area" in df.columns else [],
        "area_bucket": _agg_group_by(df, "area_bucket", area_order),
        "rooms": _agg_group(df, "room_total"),
        "building_age_bucket": [],
        "renovation": _agg_group(df, "renovation_name"),
        "building_type": _agg_group(df, "building_type_name"),
        "floor_category": _agg_group(df, "floor_category"),
        "elevator": _agg_group(df, "elevator_name"),
        "subway": _agg_group(df, "subway_name"),
        "correlation": _agg_corr(df, cols[:top_n_corr]),
        "scatter_area_price": _agg_scatter(df),
        "geo_points": _agg_geo(df),
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 3),
    }
    # 楼龄分箱
    if "building_age" in df.columns:
        b = df.copy()
        b["age_bucket"] = pd.cut(
            b["building_age"], bins=[-0.1, 5, 10, 15, 20, 30, 200],
            labels=["0-5年", "5-10年", "10-15年", "15-20年", "20-30年", "30年+"],
        ).astype(str)
        agg["building_age_bucket"] = _agg_group_by(
            b, "age_bucket", ["0-5年", "5-10年", "10-15年", "15-20年", "20-30年", "30年+"]
        )
    return agg


def save_aggregates(agg: dict[str, Any], version_id: str = "latest") -> dict[str, str]:
    """汇总结果落盘：JSON 快照 + SQLite agg_* 表。"""
    paths: dict[str, str] = {}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"aggregates_{version_id}.json"
    json_path.write_text(json.dumps(agg, ensure_ascii=False), encoding="utf-8")
    paths["json"] = str(json_path)

    latest = REPORT_DIR / "aggregates_latest.json"
    latest.write_text(json.dumps(agg, ensure_ascii=False), encoding="utf-8")
    paths["latest"] = str(latest)

    def _flatten(recs: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
        out = []
        for r in recs:
            row = {"agg_name": name}
            for k, v in r.items():
                row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            out.append(row)
        return out

    conn = connect()
    try:
        for key in ("district", "trend_monthly", "trend_quarterly", "area_bucket",
                    "rooms", "renovation", "building_type", "floor_category",
                    "elevator", "subway", "building_age_bucket"):
            recs = agg.get(key) or []
            if not recs:
                continue
            pd.DataFrame(_flatten(recs, key)).to_sql(
                f"agg_{key}", conn, if_exists="replace", index=False
            )
        pd.DataFrame([{"k": k, "v": json.dumps(v, ensure_ascii=False)}
                      for k, v in agg.items() if isinstance(v, (dict, list))]).to_sql(
            "agg_meta", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()
    return paths


def load_aggregates(version_id: str | None = None) -> dict[str, Any] | None:
    p = REPORT_DIR / (f"aggregates_{version_id}.json" if version_id else "aggregates_latest.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


__all__ = [
    "connect", "write_detail", "read_detail", "write_parquet", "read_parquet",
    "build_aggregates", "save_aggregates", "load_aggregates", "detail_count",
    "DETAIL_TABLE", "PROCESSED_DIR",
]
