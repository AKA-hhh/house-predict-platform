"""
价格预测模型（Price Regression Models）
========================================
统一封装四种回归器，输出**同一套指标 + 可解释性产物**：

  linear_ols : 线性回归基线，系数直接可读
  ridge      : L2 正则线性（特征标准化后拟合，系数在标准化空间可解释）
  hgb        : 直方图梯度提升树（精度主力）
  rf         : 随机森林（对照）

可解释性：
  - 全局：线性模型给系数表；树模型给置换重要性（permutation importance）
  - 局部：给出单条房源各特征对预测值的**贡献度**（线性用 coef·(x-x̄)；
    树模型用 标准化偏离 × 置换重要性 近似，并在结果中显式标注为近似）
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge

from .metrics import regression_metrics
from .registry import get_algorithm, validate_params

RANDOM_STATE = 42


# ------------------------------------------------------------ 构造估计器
def build_estimator(algo_id: str, params: dict[str, Any]):
    p = validate_params(algo_id, params)
    if algo_id == "linear_ols":
        return LinearRegression(fit_intercept=p["fit_intercept"], positive=p["positive"]), p
    if algo_id == "ridge":
        return Ridge(alpha=p["alpha"], random_state=None), p
    if algo_id == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=p["learning_rate"],
            max_iter=p["max_iter"],
            max_depth=p["max_depth"] if p["max_depth"] > 0 else None,
            min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"],
            max_bins=p["max_bins"],
            early_stopping=False,
            random_state=RANDOM_STATE,
        ), p
    if algo_id == "rf":
        return RandomForestRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            n_jobs=p["n_jobs"],
            random_state=RANDOM_STATE,
        ), p
    raise KeyError(f"未注册的价格预测算法: {algo_id}")


# ------------------------------------------------------------ 训练主流程
def train_price_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    algo_id: str,
    params: dict[str, Any] | None = None,
    features: list[str] | None = None,
    valid: pd.DataFrame | None = None,
    explain_samples: int = 4000,
    target_mode: str = "absolute",
) -> dict[str, Any]:
    """训练 + 评估 + 生成可解释性产物。返回结果字典（模型对象单独返回以便持久化）。

    target_mode
      - "absolute" : 直接预测单价 unit_price（元/㎡）
      - "relative" : 预测**相对价格** = unit_price / community_avg_price，
                     再乘回小区均价还原为单价。这样可以把"市场整体涨跌"这一
                     强趋势项从目标中剥离，显著缓解时间切分下的分布漂移。
                     参照价格依据真实成交记录计算，属于可观测的区位基准。
    """
    from ..pipeline.features import feature_matrix
    from ..pipeline.schema import TARGET_COLUMN

    t0 = time.time()
    spec = get_algorithm(algo_id)
    Xtr, feats = feature_matrix(train, features)
    Xte, _ = feature_matrix(test, feats)

    RATIO_COL = "community_avg_price"
    use_ratio = (target_mode == "relative") and (RATIO_COL in train.columns) and (RATIO_COL in test.columns)
    if target_mode == "relative" and not use_ratio:
        target_mode = "absolute"          # 缺少参照列时自动降级

    def _target(d: pd.DataFrame) -> np.ndarray:
        y = d[TARGET_COLUMN].astype(float).values
        if not use_ratio:
            return y
        base = np.maximum(d[RATIO_COL].astype(float).values, 1.0)
        return np.clip(y / base, 0.10, 8.0)

    ytr = _target(train)
    yte = _target(test)
    base_te = (np.maximum(test[RATIO_COL].astype(float).values, 1.0) if use_ratio else None)
    base_tr = (np.maximum(train[RATIO_COL].astype(float).values, 1.0) if use_ratio else None)

    model, p = build_estimator(algo_id, params or {})

    # 线性模型：岭回归在标准化空间拟合（alpha 才有意义），线性基线保持原始空间
    scaler = None
    if algo_id == "ridge":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr)
        Xtr_fit = scaler.transform(Xtr)
        Xte_fit = scaler.transform(Xte)
    else:
        Xtr_fit, Xte_fit = Xtr.values, Xte.values

    t_fit = time.time()
    model.fit(Xtr_fit, ytr)
    fit_seconds = round(time.time() - t_fit, 2)

    pred_tr = model.predict(Xtr_fit)
    pred_te = model.predict(Xte_fit)

    # 还原到单价尺度，保证所有模型指标口径一致、可横向比较
    ytr_raw = train[TARGET_COLUMN].astype(float).values
    yte_raw = test[TARGET_COLUMN].astype(float).values
    pred_tr_raw = pred_tr * base_tr if use_ratio else pred_tr
    pred_te_raw = pred_te * base_te if use_ratio else pred_te

    result: dict[str, Any] = {
        "algo_id": algo_id,
        "algo_name": spec["name"],
        "task": "price_regression",
        "is_linear": spec["interpretable"],
        "params": p,
        "target_mode": target_mode,
        "features": feats,
        "n_features": len(feats),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "fit_seconds": fit_seconds,
        "metrics": {
            "train": regression_metrics(ytr_raw, pred_tr_raw),
            "test": regression_metrics(yte_raw, pred_te_raw),
        },
        "metrics_in_target_space": {
            "test": regression_metrics(yte, pred_te),
        },
    }
    if valid is not None and len(valid):
        Xva, _ = feature_matrix(valid, feats)
        Xva_fit = scaler.transform(Xva) if scaler is not None else Xva.values
        pv = model.predict(Xva_fit)
        base_va = (np.maximum(valid[RATIO_COL].astype(float).values, 1.0) if use_ratio else None)
        result["metrics"]["valid"] = regression_metrics(
            valid[TARGET_COLUMN].astype(float).values,
            pv * base_va if use_ratio else pv,
        )

    # ------------------------------------------------ 可解释性：全局
    coef_table: list[dict[str, Any]] = []
    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        intercept = float(np.ravel(getattr(model, "intercept_", 0.0))[0])
        coef_table = sorted(
            [{"feature": f, "coef": round(float(c), 4)} for f, c in zip(feats, coefs)],
            key=lambda r: abs(r["coef"]), reverse=True,
        )
        result["intercept"] = round(intercept, 4)
        result["coef_space"] = "standardized" if scaler is not None else "raw"
    else:
        n = min(len(Xte), explain_samples)
        idx = np.random.default_rng(RANDOM_STATE).choice(len(Xte), n, replace=False)
        try:
            pi = permutation_importance(
                model, Xte_fit[idx], yte[idx], n_repeats=5,
                random_state=RANDOM_STATE, n_jobs=-1, scoring="neg_mean_absolute_error",
            )
            coef_table = sorted(
                [{"feature": f, "importance": round(float(m), 2), "std": round(float(s), 2)}
                 for f, m, s in zip(feats, pi.importances_mean, pi.importances_std)],
                key=lambda r: r["importance"], reverse=True,
            )
        except Exception as exc:  # 置换重要性失败不阻断训练
            coef_table = [{"feature": f, "importance": 0.0, "std": 0.0, "error": str(exc)} for f in feats]
        result["importance_method"] = "permutation_importance(neg_MAE)"

    result["explain_table"] = coef_table[:30]

    # ------------------------------------------------ 预测区间（基于测试残差，单价尺度）
    resid = yte_raw - pred_te_raw
    sigma = float(np.std(resid))
    result["residual_sigma"] = round(sigma, 2)
    result["residual_quantiles"] = {
        "p05": round(float(np.quantile(resid, 0.05)), 2),
        "p50": round(float(np.quantile(resid, 0.50)), 2),
        "p95": round(float(np.quantile(resid, 0.95)), 2),
    }

    # ------------------------------------------------ 误差明细（用于失败案例分析）
    err = np.abs(pred_te_raw - yte_raw)
    order = np.argsort(err)[::-1]
    sample_cols = [c for c in ("house_id", "district_name", "area", TARGET_COLUMN,
                               "total_price", "building_age", "build_year", "trade_time")
                   if c in test.columns]
    worst = test.iloc[order[:200]][sample_cols].copy()
    worst["pred"] = pred_te_raw[order[:200]].round(1)
    worst["abs_err"] = err[order[:200]].round(1)
    worst["ape(%)"] = (err[order[:200]] / np.maximum(yte_raw[order[:200]], 1) * 100).round(2)
    result["worst_cases"] = worst.head(20).to_dict(orient="records")
    result["residual_hist"] = _residual_hist(resid)
    result["pred_vs_true"] = _pred_vs_true(yte_raw, pred_te_raw)
    result["seconds"] = round(time.time() - t0, 2)

    bundle = {
        "algo_id": algo_id, "params": p, "features": feats,
        "model": model, "scaler": scaler,
        "target_mode": target_mode,
        "ratio_col": RATIO_COL if use_ratio else None,
        "train_mean": Xtr.mean(numeric_only=True).to_dict(),
        "train_std": Xtr.std(numeric_only=True).replace(0, 1).to_dict(),
        "residual_sigma": sigma,
        "coef_table": coef_table,
        "is_linear": hasattr(model, "coef_"),
        "coef_space": "standardized" if scaler is not None else "raw",
        "target_mean": float(ytr.mean()),
        "target_median": float(np.median(ytr)),
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"result": result, "bundle": bundle}


# ------------------------------------------------------------ 辅助产物
def _residual_hist(resid: np.ndarray, bins: int = 40) -> list[dict[str, Any]]:
    lo, hi = np.quantile(resid, 0.005), np.quantile(resid, 0.995)
    r = resid[(resid >= lo) & (resid <= hi)]
    cnt, edges = np.histogram(r, bins=bins)
    return [{"bin": round(float((edges[i] + edges[i + 1]) / 2), 0),
             "count": int(cnt[i])} for i in range(len(cnt))]


def _pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, n: int = 900) -> list[dict[str, Any]]:
    idx = np.random.default_rng(3).choice(len(y_true), min(n, len(y_true)), replace=False)
    return [{"true": round(float(y_true[i]), 0), "pred": round(float(y_pred[i]), 0)} for i in idx]


# ------------------------------------------------------------ 单样本解释
def explain_prediction(bundle: dict[str, Any], x: pd.Series) -> dict[str, Any]:
    """给出单条房源的预测值与各特征贡献度（局部可解释）。"""
    feats = bundle["features"]
    model, scaler = bundle["model"], bundle["scaler"]
    xv = pd.to_numeric(x.reindex(feats), errors="coerce").fillna(0.0).astype(float)

    x_in = scaler.transform(xv.values.reshape(1, -1)) if scaler is not None else xv.values.reshape(1, -1)
    pred_raw = float(model.predict(x_in)[0])

    # 相对价格模型：预测值乘回小区均价才是单价；贡献度同样按该系数折算到元/㎡
    ratio_mode = bundle.get("target_mode") == "relative"
    scale = 1.0
    if ratio_mode:
        rc = bundle.get("ratio_col") or "community_avg_price"
        scale = float(xv.get(rc, 0.0) or 0.0)
        if scale <= 0:
            scale = float(pd.Series(bundle["train_mean"]).get(rc, 1.0) or 1.0)
    pred = pred_raw * scale

    contributions: list[dict[str, Any]] = []
    if bundle["is_linear"]:
        mean = pd.Series(bundle["train_mean"])
        coefs = np.ravel(model.coef_)
        for f, c in zip(feats, coefs):
            delta = float(xv[f] - mean.get(f, 0.0))
            contributions.append({
                "feature": f, "value": round(float(xv[f]), 4),
                "coef": round(float(c), 4),
                "contribution": round(float(c * delta) * scale, 2),
                "direction": "↑" if c * delta > 0 else ("↓" if c * delta < 0 else "—"),
            })
        method = ("线性贡献 = 系数 × (本房源取值 − 训练集均值)"
                  + ("，再乘以小区均价折算到元/㎡（相对价格模型）" if ratio_mode else ""))
    else:
        imp = {r["feature"]: r.get("importance", 0.0) for r in bundle["coef_table"]}
        mean = pd.Series(bundle["train_mean"])
        std = pd.Series(bundle["train_std"])
        total_imp = sum(abs(v) for v in imp.values()) or 1.0
        for f in feats:
            z = float((xv[f] - mean.get(f, 0.0)) / (std.get(f, 1.0) or 1.0))
            contrib = float(imp.get(f, 0.0) / total_imp * z * abs(pred) * 0.5)
            contributions.append({
                "feature": f, "value": round(float(xv[f]), 4),
                "importance": round(float(imp.get(f, 0.0)), 2),
                "contribution": round(contrib, 2),
                "direction": "↑" if contrib > 0 else ("↓" if contrib < 0 else "—"),
            })
        method = "树模型近似归因 = 标准化偏离 × 全局置换重要性（仅作方向性参考）"

    contributions.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    up = sum(c["contribution"] for c in contributions if c["contribution"] > 0)
    down = sum(c["contribution"] for c in contributions if c["contribution"] < 0)

    sigma = bundle.get("residual_sigma", 0.0)
    return {
        "prediction": round(pred, 2),
        "prediction_wan": round(pred * float(x.get("area", 0) or 0) / 10000, 2),
        "interval_90": [round(pred - 1.645 * sigma, 2), round(pred + 1.645 * sigma, 2)],
        "top_contributions": contributions[:12],
        "positive_sum": round(up, 2),
        "negative_sum": round(down, 2),
        "method": method,
        "algo_id": bundle["algo_id"],
        "target_mode": bundle.get("target_mode", "absolute"),
        "ratio_prediction": round(pred_raw, 4) if ratio_mode else None,
        "ratio_base": round(scale, 2) if ratio_mode else None,
    }


__all__ = ["build_estimator", "train_price_model", "explain_prediction"]
