"""
Stage 2 · 数据清洗（Clean）
=============================
四类清洗动作，全部**参数化**，便于做对比实验：

  1. 记录级过滤 : 必填字段缺失、重复记录、目标变量非法的记录直接剔除
  2. 类型/取值规范化 : 数值列越界置 NaN、类别编码转可读标签
  3. 缺失值处理 : drop / mean / median / mode / ffill / 常数填充（按列指定）
  4. 异常值处理 : iqr（1.5~3.0 倍四分位距）/ zscore / range（业务区间）/ none

每一步都会返回结构化日志（处理了哪些列、删了多少行、替换了多少值），
写入数据版本记录，保证清洗过程**可复现、可审计**。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    BUILDING_STRUCTURE_MAP, BUILDING_TYPE_MAP, DISTRICT_MAP,
    LABEL_COLUMNS, OUTLIER_ACTIONS, OUTLIER_COLUMNS, RENOVATION_MAP,
    TARGET_BUSINESS_RANGE, TARGET_COLUMN, VALID_RANGES,
)

# 目标变量必须存在且合理，否则整条记录无意义
REQUIRED_COLUMNS = ["unit_price", "total_price", "area", "trade_time", "district_code"]

MISSING_STRATEGIES = ["drop", "mean", "median", "mode", "constant", "ffill"]
OUTLIER_STRATEGIES = ["iqr", "zscore", "range", "none"]


# ------------------------------------------------------------ 1. 记录过滤
def filter_records(df: pd.DataFrame, log: dict[str, Any]) -> pd.DataFrame:
    n0 = len(df)

    # (a) 必填字段缺失
    before = len(df)
    df = df.dropna(subset=[c for c in REQUIRED_COLUMNS if c in df.columns])
    log["drop_missing_required"] = before - len(df)

    # (b) 完全重复
    before = len(df)
    df = df.drop_duplicates()
    log["drop_exact_duplicate"] = before - len(df)

    # (c) 同一 house_id 重复（保留最后一条成交记录）
    if "house_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["house_id"], keep="last")
        log["drop_duplicate_house_id"] = before - len(df)

    # (d) 目标变量非正
    before = len(df)
    df = df[df[TARGET_COLUMN] > 0]
    log["drop_nonpositive_target"] = before - len(df)

    # (e) 目标变量超出业务硬区间（统计方法对右偏分布的低端漏检，需业务兜底）
    lo, hi = TARGET_BUSINESS_RANGE
    before = len(df)
    df = df[(df[TARGET_COLUMN] >= lo) & (df[TARGET_COLUMN] <= hi)]
    log["drop_target_out_of_business_range"] = before - len(df)
    log["target_business_range"] = [lo, hi]

    log["rows_after_filter"] = len(df)
    log["rows_dropped_total"] = n0 - len(df)
    return df.reset_index(drop=True)


# ------------------------------------------------------- 2. 取值规范化
def normalize_values(df: pd.DataFrame, log: dict[str, Any]) -> pd.DataFrame:
    # 地理坐标越界置空（北京大致经纬度包围盒，宽松边界）
    if {"lng", "lat"}.issubset(df.columns):
        bad = (
            (df["lng"] < 115.0) | (df["lng"] > 118.0) |
            (df["lat"] < 39.0) | (df["lat"] > 41.5)
        )
        log["invalid_geo_nulled"] = int(bad.sum())
        for c in ("lng", "lat"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df[c] = df[c].where(~bad, np.nan)

    # 类别编码 -> 可读标签
    if "district_code" in df.columns:
        d = pd.to_numeric(df["district_code"], errors="coerce")
        df["district_code"] = d
        df["district_name"] = d.map(DISTRICT_MAP).fillna("未知区域")
    if "building_type" in df.columns:
        df["building_type_name"] = df["building_type"].map(BUILDING_TYPE_MAP).fillna("未知")
    if "building_structure" in df.columns:
        df["building_structure_name"] = (
            df["building_structure"].map(BUILDING_STRUCTURE_MAP).fillna("未知结构")
        )
    if "renovation" in df.columns:
        df["renovation_name"] = df["renovation"].map(RENOVATION_MAP).fillna("未知")
    if "elevator" in df.columns:
        df["elevator_name"] = df["elevator"].map({1: "有电梯", 0: "无电梯"}).fillna("未知")
    if "subway" in df.columns:
        df["subway_name"] = df["subway"].map({1: "近地铁", 0: "不近地铁"}).fillna("未知")
    if "five_years" in df.columns:
        df["five_years_name"] = df["five_years"].map({1: "满五年", 0: "未满五年"}).fillna("未知")

    # 成交时间解析
    if "trade_time" in df.columns:
        parsed = pd.to_datetime(df["trade_time"], errors="coerce")
        bad = int(parsed.isna().sum() - df["trade_time"].isna().sum())
        log["unparsable_trade_time"] = max(bad, 0)
        df["trade_time"] = parsed
        df = df[df["trade_time"].notna()].reset_index(drop=True)

    log["label_columns_created"] = [c for c in LABEL_COLUMNS if c in df.columns]
    return df


# -------------------------------------------------------- 3. 缺失值处理
def handle_missing(
    df: pd.DataFrame,
    strategy: str = "median",
    per_column: dict[str, str] | None = None,
    drop_threshold: float = 0.6,
    log: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """缺失值处理。

    strategy      : 全局默认策略 drop|mean|median|mode|constant|ffill
    per_column    : 列级覆盖，如 {"community_avg_price": "median"}
    drop_threshold: 缺失率高于该阈值的**特征列**整列剔除（默认 0.6）
    """
    log = log if log is not None else {}
    per_column = per_column or {}
    n0 = len(df)

    # (a) 高缺失率列剔除（保护目标列与标识列）
    protected = {TARGET_COLUMN, "house_id", "trade_time"}
    dropped_cols = []
    for col in list(df.columns):
        if col in protected:
            continue
        rate = float(df[col].isna().mean())
        if rate > drop_threshold and rate > 0:
            df = df.drop(columns=[col])
            dropped_cols.append({"column": col, "missing_rate": round(rate, 4)})
    log["columns_dropped_high_missing"] = dropped_cols

    # (b) 全局策略为 drop 时，直接删含缺失的行
    if strategy == "drop" and not per_column:
        df = df.dropna()
        log["rows_dropped_missing"] = n0 - len(df)
        log["filled_cells"] = 0
        return df.reset_index(drop=True)

    # (c) 列级填充
    filled = 0
    detail: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        if miss == 0:
            continue
        strat = per_column.get(col)
        if strat is None:
            if pd.api.types.is_numeric_dtype(s):
                strat = strategy if strategy in ("mean", "median", "ffill", "constant") else "median"
            else:
                strat = "mode" if strategy in ("mode", "constant", "ffill") else "mode"
        if strat == "drop":
            continue

        before = miss
        if strat == "mean" and pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(s.mean())
        elif strat == "median" and pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(s.median())
        elif strat == "mode":
            m = s.mode(dropna=True)
            df[col] = s.fillna(m.iloc[0]) if len(m) else s.fillna("未知")
        elif strat == "ffill":
            df[col] = s.ffill().bfill()
        elif strat == "constant":
            df[col] = s.fillna(0 if pd.api.types.is_numeric_dtype(s) else "未知")
        else:
            # 未知策略 -> 数值取中位数，类别取众数
            df[col] = s.fillna(s.median()) if pd.api.types.is_numeric_dtype(s) else s.fillna("未知")

        after = int(df[col].isna().sum())
        filled += before - after
        detail[col] = {"strategy": strat, "filled": before - after, "missing_rate": round(before / n0, 4)}

    # (d) 仍残留缺失的行（列级指定 drop 的列）直接删除
    if df.isna().any().any():
        before = len(df)
        df = df.dropna()
        log["rows_dropped_missing"] = before - len(df)
    else:
        log["rows_dropped_missing"] = 0

    log["fill_detail"] = detail
    log["filled_cells"] = int(filled)
    log["rows_after_missing"] = len(df)
    return df.reset_index(drop=True)


# -------------------------------------------------------- 4. 异常值处理
def handle_outliers(
    df: pd.DataFrame,
    strategy: str = "iqr",
    iqr_k: float = 1.5,
    zscore_threshold: float = 3.0,
    columns: list[str] | None = None,
    action: str = "winsorize",
    log: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """异常值处理。

    strategy : iqr | zscore | range | none
      - iqr    : 超出 [Q1 - k*IQR, Q3 + k*IQR] 视为异常
      - zscore : |z| > threshold 视为异常
      - range  : 超出 schema.VALID_RANGES 业务区间视为异常
    columns  : 检测列。默认只对**业务列**（schema.OUTLIER_COLUMNS）检测，而不是所有数值列
               —— 像 days_on_market / followers 这类重尾计数量，用 IQR 截断会把整列压向
               中位数，反而破坏信息（详见 docs/05 问题清单 A11）。
    action   : 处置动作
      - winsorize   : 越界值封顶/保底到边界（默认，保留样本且保留排序信息）
      - clip_median : 越界值置为列中位数（更激进，会损失排序信息）
      - drop        : 直接删除越界记录
    """
    log = log if log is not None else {}
    if strategy == "none":
        log.update({"strategy": "none", "action": "none",
                    "outliers_detected": 0, "rows_dropped": 0, "affected_columns": {}})
        return df

    if action not in OUTLIER_ACTIONS:
        raise ValueError(f"不支持的异常值处置动作: {action} (可选 {OUTLIER_ACTIONS})")

    detail: dict[str, dict[str, Any]] = {}
    total = 0

    if columns:
        target_cols = list(columns)
    elif strategy == "range":
        target_cols = [c for c in VALID_RANGES if c in df.columns]
    else:
        target_cols = [c for c in OUTLIER_COLUMNS if c in df.columns]

    drop_mask_all = pd.Series(False, index=df.index)

    for col in target_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col]

        if strategy == "range":
            bounds = VALID_RANGES.get(col)
            if bounds is None:
                continue
            lo, hi = bounds
            mask = (s < lo) | (s > hi)
        elif strategy == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            mask = (s < lo) | (s > hi)
        elif strategy == "zscore":
            mu, sd = s.mean(), s.std()
            if not sd or pd.isna(sd):
                continue
            z = (s - mu) / sd
            mask = z.abs() > zscore_threshold
            lo, hi = mu - zscore_threshold * sd, mu + zscore_threshold * sd
        else:
            raise ValueError(f"不支持的异常值策略: {strategy}")

        n = int(mask.sum())
        if n == 0:
            continue

        if action == "winsorize":
            df[col] = s.clip(lower=lo, upper=hi).astype("float64")
            act = f"winsorize_to[{round(float(lo), 2)}, {round(float(hi), 2)}]"
        elif action == "clip_median":
            med = float(s.median())
            df[col] = s.where(~mask, med)
            act = f"clip_to_median({round(med, 2)})"
        else:  # drop
            drop_mask_all |= mask.fillna(False)
            act = "mark_for_drop"

        total += n
        detail[col] = {
            "detected": n,
            "rate": round(n / max(len(df), 1), 5),
            "lower_bound": round(float(lo), 4),
            "upper_bound": round(float(hi), 4),
            "action": act,
        }

    dropped = 0
    if action == "drop" and bool(drop_mask_all.any()):
        before = len(df)
        df = df[~drop_mask_all]
        dropped = before - len(df)

    log.update({
        "strategy": strategy,
        "action": action,
        "iqr_k": iqr_k if strategy == "iqr" else None,
        "zscore_threshold": zscore_threshold if strategy == "zscore" else None,
        "checked_columns": target_cols,
        "outliers_detected": int(total),
        "rows_dropped": int(dropped),
        "affected_columns": detail,
    })
    return df.reset_index(drop=True) if dropped else df

# ----------------------------------------------------------- 主入口
def clean(
    df: pd.DataFrame,
    missing_strategy: str = "median",
    missing_per_column: dict[str, str] | None = None,
    drop_threshold: float = 0.6,
    outlier_strategy: str = "iqr",
    iqr_k: float = 1.5,
    zscore_threshold: float = 3.0,
    outlier_action: str = "winsorize",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """串起四步清洗，返回 (干净数据, 清洗报告)。"""
    t0 = time.time()
    n_in = len(df)
    report: dict[str, Any] = {
        "params": {
            "missing_strategy": missing_strategy,
            "missing_per_column": missing_per_column or {},
            "drop_threshold": drop_threshold,
            "outlier_strategy": outlier_strategy,
            "outlier_action": outlier_action,
            "iqr_k": iqr_k,
            "zscore_threshold": zscore_threshold,
        },
        "rows_in": n_in,
    }

    filt: dict[str, Any] = {}
    df = filter_records(df, filt)
    report["filter"] = filt

    norm: dict[str, Any] = {}
    df = normalize_values(df, norm)
    report["normalize"] = norm

    miss: dict[str, Any] = {}
    df = handle_missing(df, missing_strategy, missing_per_column, drop_threshold, miss)
    report["missing"] = miss

    out: dict[str, Any] = {}
    df = handle_outliers(df, outlier_strategy, iqr_k, zscore_threshold, None,
                         outlier_action, out)
    out["target_policy"] = (
        f"目标变量 {TARGET_COLUMN} 不做统计截断（避免改写标签），"
        f"仅按业务区间 {TARGET_BUSINESS_RANGE} 硬过滤；"
        f"越界记录已在 filter_records 中剔除 "
        f"{filt.get('drop_target_out_of_business_range', 0)} 条。"
    )
    report["outlier"] = out

    report["rows_out"] = len(df)
    report["rows_dropped_total"] = n_in - len(df)
    report["seconds"] = round(time.time() - t0, 3)
    return df.reset_index(drop=True), report


__all__ = [
    "clean", "filter_records", "normalize_values", "handle_missing",
    "handle_outliers", "MISSING_STRATEGIES", "OUTLIER_STRATEGIES", "OUTLIER_ACTIONS",
]
