"""
评价指标（Metrics）
====================
回归指标：MAE / RMSE / R² / MAPE / MedAE / 带内命中率
排序指标：Precision@K / Recall@K / NDCG@K / HitRate@K / MAP / 覆盖率 / 多样性
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


# ============================================================ 回归指标
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {}

    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    medae = float(np.median(np.abs(err)))

    nz = np.abs(yt) > 1e-9
    mape = float(np.mean(np.abs(err[nz] / yt[nz])) * 100) if nz.any() else float("nan")

    # 业务口径：预测值落在真实值 ±10% 区间内的比例
    within10 = float(np.mean(np.abs(err[nz] / yt[nz]) <= 0.10) * 100) if nz.any() else float("nan")
    within20 = float(np.mean(np.abs(err[nz] / yt[nz]) <= 0.20) * 100) if nz.any() else float("nan")

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MedAE": round(medae, 2),
        "MAPE(%)": round(mape, 2),
        "Within10%(%)": round(within10, 2),
        "Within20%(%)": round(within20, 2),
        "n_samples": int(len(yt)),
    }


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_true)))


# ============================================================ 排序指标
def precision_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(recommended)[:k]
    if not top:
        return 0.0
    return sum(1 for i in top if i in relevant) / len(top)


def recall_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = list(recommended)[:k]
    return sum(1 for i in top if i in relevant) / len(relevant)


def ndcg_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    top = list(recommended)[:k]
    dcg = sum((1.0 / math.log2(i + 2)) for i, item in enumerate(top) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def hit_rate_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    return 1.0 if any(i in relevant for i in list(recommended)[:k]) else 0.0


def average_precision(recommended: Sequence, relevant: set, k: int) -> float:
    hits, score = 0, 0.0
    for i, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return float(score / min(len(relevant), k)) if relevant and k > 0 else 0.0


def coverage(all_recommended: list[list], n_items: int) -> float:
    seen = set()
    for rec in all_recommended:
        seen.update(rec)
    return round(len(seen) / max(n_items, 1), 4)


def intra_list_diversity(recommended: Sequence, item_vectors: dict) -> float:
    """推荐列表内部平均两两不相似度（1 - 余弦相似度）。"""
    items = [i for i in recommended if i in item_vectors]
    if len(items) < 2:
        return 0.0
    sims = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = item_vectors[items[i]], item_vectors[items[j]]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                sims.append(float(np.dot(a, b) / (na * nb)))
    return round(1 - float(np.mean(sims)), 4) if sims else 0.0


def ranking_report(
    per_user_recs: dict[Any, list],
    per_user_relevant: dict[Any, set],
    ks: Sequence[int] = (5, 10, 20),
    n_items: int = 0,
) -> dict[str, Any]:
    """汇总一个推荐器的离线排序指标。"""
    out: dict[str, Any] = {"n_users": len(per_user_recs)}
    for k in ks:
        p = np.mean([precision_at_k(r, per_user_relevant[u], k)
                     for u, r in per_user_recs.items()]) if per_user_recs else 0.0
        rc = np.mean([recall_at_k(r, per_user_relevant[u], k)
                      for u, r in per_user_recs.items()]) if per_user_recs else 0.0
        nd = np.mean([ndcg_at_k(r, per_user_relevant[u], k)
                      for u, r in per_user_recs.items()]) if per_user_recs else 0.0
        hr = np.mean([hit_rate_at_k(r, per_user_relevant[u], k)
                      for u, r in per_user_recs.items()]) if per_user_recs else 0.0
        mp = np.mean([average_precision(r, per_user_relevant[u], k)
                      for u, r in per_user_recs.items()]) if per_user_recs else 0.0
        out[f"Precision@{k}"] = round(float(p), 4)
        out[f"Recall@{k}"] = round(float(rc), 4)
        out[f"NDCG@{k}"] = round(float(nd), 4)
        out[f"HitRate@{k}"] = round(float(hr), 4)
        out[f"MAP@{k}"] = round(float(mp), 4)
    if n_items:
        out["Coverage"] = coverage(list(per_user_recs.values()), n_items)
    return out


__all__ = [
    "regression_metrics", "mae", "precision_at_k", "recall_at_k", "ndcg_at_k",
    "hit_rate_at_k", "average_precision", "coverage", "intra_list_diversity",
    "ranking_report",
]
