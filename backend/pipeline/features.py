"""
Stage 3 · 特征构造（Feature Engineering）
==========================================
把清洗后的原始字段加工为模型可用的特征，全部为**确定性变换**（无随机、无未来信息）：

  位置类   : 到城市中心的直线距离、是否核心城区、经纬度
  房屋类   : 户型总数、单间均摊面积、楼层占比（所在层/总层）、楼龄
  品质类   : 电梯、地铁、满五年、装修、建筑类型/结构（编码）
  时间类   : 成交年/月/季度/星期（用于捕捉市场周期）
  小区类   : 小区均价及其与区域均价的比值（区位热度）

⚠️ 防泄漏说明：
  目标变量为 unit_price（单价）。total_price = unit_price × area / 10000，
  因此**不得**把 total_price / area 的派生量作为特征，否则会引入严重泄漏。
  本模块通过 ASSERT 显式禁止这类派生列进入特征集。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .schema import MODEL_NUMERIC_FEATURES, RECOMMEND_FEATURES, TARGET_COLUMN

# 北京市中心参考点（天安门），用于计算直线距离
CITY_CENTER = (116.397428, 39.90923)
CORE_DISTRICTS = {"东城区", "西城区", "朝阳区", "海淀区", "丰台区", "石景山区"}

# 严禁作为特征的列（目标派生量 / 标识 / 原文）
FORBIDDEN_FEATURES = {"unit_price", "total_price", "total_price_calc", "price_per_area"}


def _haversine(lng: pd.Series, lat: pd.Series) -> pd.Series:
    """向量化球面距离（km）。"""
    r = 6371.0
    lon1, lat1 = np.radians(CITY_CENTER[0]), np.radians(CITY_CENTER[1])
    lon2, lat2 = np.radians(lng.astype(float)), np.radians(lat.astype(float))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_floor(df: pd.DataFrame, log: dict[str, Any]) -> pd.DataFrame:
    """解析 floor_raw（形如 '高 26' / '中 12' / '低 3'）-> 楼层类别 + 所在层 + 总层 + 占比。"""
    if "floor_raw" not in df.columns:
        return df
    raw = df["floor_raw"].astype(str).str.strip()
    parts = raw.str.extract(r"^(?P<cat>[低中高])?\s*(?P<num>\d+)?$")
    cat = parts["cat"].fillna("未知")
    total_floors = pd.to_numeric(parts["num"], errors="coerce")

    df["floor_category"] = cat.map({"低": "低楼层", "中": "中楼层", "高": "高楼层"}).fillna("未知")
    df["floor_total"] = total_floors

    # 所在层：依据类别与总层的对应区间估算（低=1/1，中=总层一半，高=总层）
    frac = cat.map({"低": 1 / 6, "中": 0.5, "高": 5 / 6}).fillna(0.5)
    df["floor_number"] = (total_floors * frac).round()
    df["floor_ratio"] = (frac).round(4)

    log["floor_parsed"] = int(parts["cat"].notna().sum())
    log["floor_unknown"] = int((cat == "未知").sum())
    log["floor_total_missing"] = int(total_floors.isna().sum())
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构造全部特征列，返回 (含特征的数据, 特征报告)。"""
    t0 = time.time()
    log: dict[str, Any] = {}
    n0 = len(df)

    # ---------------------------------------------------- 1. 楼层解析
    df = parse_floor(df, log)

    # ---------------------------------------------------- 2. 户型
    for c in ("living_room", "drawing_room", "kitchen", "bath_room"):
        if c not in df.columns:
            df[c] = 0
    df["room_total"] = df[["living_room", "drawing_room", "kitchen", "bath_room"]].sum(axis=1)
    df["area_per_room"] = (df["area"] / df["room_total"].replace(0, np.nan)).round(2)
    df["bedroom_ratio"] = (df["living_room"] / df["room_total"].replace(0, np.nan)).round(4)
    log["room_features"] = ["room_total", "area_per_room", "bedroom_ratio"]

    # ---------------------------------------------------- 3. 楼龄 / 时间
    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df["trade_year"] = df["trade_time"].dt.year
    df["trade_month"] = df["trade_time"].dt.month
    df["trade_quarter"] = df["trade_time"].dt.quarter
    df["trade_dow"] = df["trade_time"].dt.dayofweek
    df["trade_ym"] = df["trade_time"].dt.strftime("%Y-%m")

    if "build_year" in df.columns:
        by = pd.to_numeric(df["build_year"], errors="coerce")
        by = by.where((by >= 1900) & (by <= 2020))
        df["build_year"] = by
        df["building_age"] = (df["trade_year"] - by).clip(lower=0)
        # 建筑年代缺失 -> 用区域中位数回填
        if "district_name" in df.columns:
            med = df.groupby("district_name")["building_age"].transform("median")
            df["building_age"] = df["building_age"].fillna(med)
        df["building_age"] = df["building_age"].fillna(df["building_age"].median())
    log["time_features"] = ["trade_year", "trade_month", "trade_quarter", "trade_dow", "building_age"]

    # ---------------------------------------------------- 4. 位置
    if {"lng", "lat"}.issubset(df.columns):
        df["dist_to_center"] = _haversine(df["lng"], df["lat"]).round(3)
        df["is_core_district"] = df["district_name"].isin(CORE_DISTRICTS).astype(int)
        log["geo_features"] = ["dist_to_center", "is_core_district"]

    # ---------------------------------------------------- 5. 小区热度
    if "community_avg_price" in df.columns:
        cap = pd.to_numeric(df["community_avg_price"], errors="coerce")
        df["community_avg_price"] = cap
        if "district_name" in df.columns:
            dmean = df.groupby("district_name")["community_avg_price"].transform("mean")
            df["community_vs_district"] = (cap / dmean.replace(0, np.nan)).round(4)
        df["community_avg_price"] = df["community_avg_price"].fillna(
            df.groupby("district_name")["community_avg_price"].transform("median")
            if "district_name" in df.columns else df["community_avg_price"].median()
        )
        log["community_features"] = ["community_avg_price", "community_vs_district"]

    # ---------------------------------------------------- 6. 类别编码
    code_map = {
        "floor_category": "floor_category_code",
        "building_type_name": "building_type_code",
        "building_structure_name": "building_structure_code",
        "renovation_name": "renovation_code",
    }
    made = []
    for src, dst in code_map.items():
        if src in df.columns:
            df[dst] = df[src].astype("category").cat.codes.replace(-1, np.nan)
            made.append(dst)
    log["encoded_categories"] = made

    # ---------------------------------------------------- 7. 面积分箱（供可视化/规则使用）
    if "area" in df.columns:
        bins = [0, 50, 70, 90, 120, 150, 200, 1e9]
        labels = ["<50", "50-70", "70-90", "90-120", "120-150", "150-200", "200+"]
        df["area_bucket"] = pd.cut(df["area"], bins=bins, labels=labels).astype(str)

    # ---------------------------------------------------- 8. 汇总
    present = [c for c in MODEL_NUMERIC_FEATURES if c in df.columns]
    leakage = sorted(set(present) & FORBIDDEN_FEATURES)
    assert not leakage, f"检测到特征泄漏列: {leakage}"

    log["model_features"] = present
    log["n_features"] = len(present)
    log["forbidden_check"] = "passed"
    log["rows_in"] = n0
    log["rows_out"] = len(df)
    log["seconds"] = round(time.time() - t0, 3)
    return df, log


def feature_matrix(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """抽取建模特征矩阵，缺失值以 0 兜底（清洗阶段应已基本填满）。"""
    feats = [c for c in (features or MODEL_NUMERIC_FEATURES) if c in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, feats


def recommend_frame(df: pd.DataFrame) -> pd.DataFrame:
    """构造推荐模块使用的特征视图。"""
    cols = [c for c in RECOMMEND_FEATURES if c in df.columns]
    extra = [c for c in ("house_id", "total_price", "area_bucket", "dist_to_center", "trade_ym") if c in df.columns]
    return df[sorted(set(cols + extra))].copy()


__all__ = [
    "build_features", "feature_matrix", "recommend_frame",
    "parse_floor", "CITY_CENTER", "FORBIDDEN_FEATURES",
]
