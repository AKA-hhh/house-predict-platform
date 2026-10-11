"""
Stage 4 · 数据划分（Split）
============================
提供两种划分协议，用于支撑不同的对比实验：

  random : 随机划分（常见但不严谨，会因同一小区/同一时期样本同时进入训练与测试而高估性能）
  time   : **按成交时间切分**（训练集 = 早期，测试集 = 后期），贴近真实预测场景

同时支持 3-way（train / valid / test）与 k 折交叉验证分组标签的生成。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .schema import TARGET_COLUMN


def split_random(
    df: pd.DataFrame,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_size)
    n_valid = int(len(df) * valid_size)

    test_idx = idx[:n_test]
    valid_idx = idx[n_test:n_test + n_valid] if n_valid else np.array([], dtype=int)
    train_idx = idx[n_test + n_valid:]

    meta = {
        "protocol": "random",
        "seed": seed,
        "test_size": test_size,
        "valid_size": valid_size,
        "train_rows": int(len(train_idx)),
        "valid_rows": int(len(valid_idx)),
        "test_rows": int(len(test_idx)),
    }
    tr = df.iloc[train_idx].reset_index(drop=True)
    te = df.iloc[test_idx].reset_index(drop=True)
    va = df.iloc[valid_idx].reset_index(drop=True) if n_valid else None
    return tr, te, va, meta


def split_time(
    df: pd.DataFrame,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    time_col: str = "trade_time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    if time_col not in df.columns:
        raise KeyError(f"时间划分需要列 {time_col}")
    d = df.sort_values(time_col).reset_index(drop=True)
    n = len(d)
    n_test = int(n * test_size)
    n_valid = int(n * valid_size)
    n_train = n - n_test - n_valid

    tr = d.iloc[:n_train].reset_index(drop=True)
    va = d.iloc[n_train:n_train + n_valid].reset_index(drop=True) if n_valid else None
    te = d.iloc[n_train + n_valid:].reset_index(drop=True)

    meta = {
        "protocol": "time",
        "time_col": time_col,
        "test_size": test_size,
        "valid_size": valid_size,
        "train_rows": int(len(tr)),
        "valid_rows": int(len(va)) if va is not None else 0,
        "test_rows": int(len(te)),
        "train_period": [str(tr[time_col].min()), str(tr[time_col].max())],
        "test_period": [str(te[time_col].min()), str(te[time_col].max())],
    }
    return tr, te, va, meta


def split(
    df: pd.DataFrame,
    protocol: str = "time",
    test_size: float = 0.2,
    valid_size: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    t0 = time.time()
    if protocol == "random":
        tr, te, va, meta = split_random(df, test_size, valid_size, seed)
    elif protocol == "time":
        tr, te, va, meta = split_time(df, test_size, valid_size)
    else:
        raise ValueError(f"不支持的划分协议: {protocol} (可选 random / time)")
    meta["seconds"] = round(time.time() - t0, 3)
    meta["target"] = TARGET_COLUMN
    return tr, te, va, meta


def kfold_groups(df: pd.DataFrame, k: int = 5, protocol: str = "random", seed: int = 42) -> np.ndarray:
    """生成 k 折标签（按顺序切，time 协议下即为时序滚动验证）。"""
    n = len(df)
    if protocol == "time":
        order = np.arange(n)
    else:
        order = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, dtype=int)
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        fold[order[edges[i]:edges[i + 1]]] = i
    return fold


__all__ = ["split", "split_random", "split_time", "kfold_groups"]
