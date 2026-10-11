"""
训练与推理编排（Trainer）
==========================
- 价格模型：训练 -> 评估 -> 产物落盘（joblib）-> 登记到模型仓库（models/registry.json）
- 推荐模型：按（数据版本, 算法, 参数）缓存，避免每次请求重建索引
- 统一对外暴露 predict / recommend / similar 接口
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..pipeline.store import DETAIL_TABLE, load_aggregates, read_detail, read_parquet
from ..pipeline.versioning import now_str, versions
from ..settings import MODEL_DIR, PROCESSED_DIR
from .price_models import explain_prediction, train_price_model
from .recommend import build_interaction_log, build_recommender
from .registry import get_algorithm, validate_params

MODEL_REGISTRY = MODEL_DIR / "registry.json"
_lock = threading.Lock()
_recommend_cache: dict[str, Any] = {}
_data_cache: dict[str, pd.DataFrame] = {}


# ============================================================ 模型仓库
def _load_model_index() -> list[dict[str, Any]]:
    if MODEL_REGISTRY.exists():
        try:
            return json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def _save_model_index(items: list[dict[str, Any]]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def register_model(rec: dict[str, Any]) -> None:
    with _lock:
        idx = [i for i in _load_model_index() if i.get("model_id") != rec["model_id"]]
        idx.insert(0, rec)
        _save_model_index(idx)


def list_models(task: str | None = None) -> list[dict[str, Any]]:
    idx = _load_model_index()
    return [m for m in idx if not task or m.get("task") == task]


def get_model_record(model_id: str) -> dict[str, Any]:
    for m in _load_model_index():
        if m["model_id"] == model_id:
            return m
    raise KeyError(f"模型不存在: {model_id}")


def delete_model(model_id: str) -> None:
    with _lock:
        idx = [i for i in _load_model_index() if i.get("model_id") != model_id]
        _save_model_index(idx)
    p = MODEL_DIR / f"{model_id}.joblib"
    if p.exists():
        p.unlink()


def load_bundle(model_id: str) -> dict[str, Any]:
    p = MODEL_DIR / f"{model_id}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"模型文件缺失: {p}")
    return joblib.load(p)


# ============================================================ 数据加载
def load_version_data(version_id: str, use_sqlite: bool = False) -> pd.DataFrame:
    """加载某数据版本的清洗+特征后全量数据（进程内缓存）。"""
    key = f"{version_id}:{use_sqlite}"
    if key in _data_cache:
        return _data_cache[key]
    if use_sqlite:
        df = read_detail(DETAIL_TABLE)
    else:
        p = PROCESSED_DIR / f"{version_id}.parquet"
        if p.exists():
            df = read_parquet(p)
        else:
            df = read_detail(DETAIL_TABLE)
    for c in ("trade_time",):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    _data_cache[key] = df
    return df


def load_split(version_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    d = MODEL_DIR / version_id
    tr = read_parquet(d / "train.parquet")
    te = read_parquet(d / "test.parquet")
    va = read_parquet(d / "valid.parquet") if (d / "valid.parquet").exists() else None
    for x in (tr, te, va):
        if x is not None and "trade_time" in x.columns:
            x["trade_time"] = pd.to_datetime(x["trade_time"], errors="coerce")
    return tr, te, va


def clear_caches() -> None:
    _recommend_cache.clear()
    _data_cache.clear()


# ============================================================ 价格模型训练
def train_price(
    algo_id: str,
    params: dict[str, Any] | None = None,
    version_id: str | None = None,
    features: list[str] | None = None,
    tags: str = "",
    target_mode: str = "absolute",
    feature_profile: str = "full",
) -> dict[str, Any]:
    """训练一个价格预测模型并登记。

    feature_profile : full | no_time | no_community | compact，见 schema.FEATURE_PROFILES
    target_mode     : absolute | relative（相对小区均价建模）
    """
    v = versions.active_version() if version_id is None else versions.get(version_id)
    if v is None:
        raise RuntimeError("尚无可用的数据版本，请先运行数据流水线")
    vid = v["version_id"]

    from ..pipeline.schema import FEATURE_PROFILES, resolve_feature_profile
    if features is None:
        features = resolve_feature_profile(feature_profile)

    tr, te, va = load_split(vid)
    out = train_price_model(tr, te, algo_id, params or {}, features, va,
                            target_mode=target_mode)
    result, bundle = out["result"], out["bundle"]

    model_id = f"{algo_id}-{vid.split('-')[-1]}-{uuid.uuid4().hex[:6]}"
    bundle["version_id"] = vid
    joblib.dump(bundle, MODEL_DIR / f"{model_id}.joblib")

    rec = {
        "model_id": model_id,
        "task": "price_regression",
        "algo_id": algo_id,
        "algo_name": result["algo_name"],
        "version_id": vid,
        "target_mode": result["target_mode"],
        "feature_profile": feature_profile,
        "feature_profile_name": FEATURE_PROFILES.get(feature_profile, {}).get("name", "自定义"),
        "params": result["params"],
        "features": result["features"],
        "n_features": result["n_features"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "metrics": result["metrics"],
        "fit_seconds": result["fit_seconds"],
        "residual_sigma": result["residual_sigma"],
        "created_at": now_str(),
        "tags": tags,
        "file": str(MODEL_DIR / f"{model_id}.joblib"),
    }
    register_model(rec)
    result["model_id"] = model_id
    result["version_id"] = vid
    return result


# ============================================================ 价格预测
def predict_price(model_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    bundle = load_bundle(model_id)
    import numpy as np

    x = pd.Series(dtype=float)
    for f in bundle["features"]:
        val = inputs.get(f)
        if val is None or val == "":
            val = bundle["train_mean"].get(f, 0.0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = bundle["train_mean"].get(f, 0.0)
        x[f] = val

    # 若前端只给了原始字段，这里做一次最小推导，避免 0 值污染
    if "area" in bundle["features"] and float(inputs.get("area") or 0) > 0:
        area = float(inputs["area"])
        if "room_total" in bundle["features"] and not inputs.get("room_total"):
            rooms = float(inputs.get("living_room") or 2) + float(inputs.get("drawing_room") or 1)
            x["room_total"] = rooms
        if "building_age" in bundle["features"] and inputs.get("building_age") in (None, ""):
            by = inputs.get("build_year")
            if by:
                x["building_age"] = max(float(inputs.get("trade_year") or 2018) - float(by), 0)
        if "trade_year" in bundle["features"] and not inputs.get("trade_year"):
            x["trade_year"] = 2017.0
            x["trade_month"] = 6.0
            x["trade_quarter"] = 2.0

    exp = explain_prediction(bundle, x)
    exp.update({
        "model_id": model_id,
        "version_id": bundle.get("version_id"),
        "residual_sigma": bundle.get("residual_sigma"),
        "algo_id": bundle["algo_id"],
    })
    return exp


# ============================================================ 推荐
def get_recommender(algo_id: str, version_id: str | None = None,
                    params: dict[str, Any] | None = None, refresh: bool = False):
    v = versions.active_version() if version_id is None else versions.get(version_id)
    if v is None:
        raise RuntimeError("尚无可用的数据版本")
    vid = v["version_id"]
    p = validate_params(algo_id, params or {})
    key = f"{vid}|{algo_id}|{json.dumps(p, sort_keys=True)}"
    if not refresh and key in _recommend_cache:
        return _recommend_cache[key], vid, p

    df = load_version_data(vid)
    log_pack = None
    if algo_id in ("item_cf", "hybrid") and p.get("n_users"):
        log_pack = build_interaction_log(df, p.get("n_users", 800), p.get("n_interactions", 12),
                                         p.get("seed", 42))
    rec = build_recommender(
        algo_id, df, p,
        log=(log_pack or {}).get("log"),
        log_stats=(log_pack or {}).get("stats"),
    )
    _recommend_cache[key] = rec
    return rec, vid, p


def recommend(algo_id: str, profile: dict[str, Any], top_k: int = 10,
              version_id: str | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.time()
    profile = dict(profile or {})
    params = dict(params or {})
    # 业务规则层：当调用方给出了**显式约束**（预算/面积/户型/区域/地铁/电梯）时，
    # 自动开启硬过滤，保证推荐结果满足用户已经明确提出的条件。
    explicit = ("budget_min", "budget_max", "area_min", "area_max", "rooms",
                "district", "subway_required", "elevator_required")
    if any(profile.get(k) for k in explicit) and "hard_filter" not in params:
        params["hard_filter"] = True

    rec, vid, p = get_recommender(algo_id, version_id, params)
    profile["top_k"] = top_k
    items = rec.recommend(profile, top_k=top_k)
    return {
        "algo_id": algo_id,
        "algo_name": get_algorithm(algo_id)["name"],
        "version_id": vid,
        "params": p,
        "profile": profile,
        "count": len(items),
        "items": items,
        "seconds": round(time.time() - t0, 3),
    }


def similar_items(house_id: str, top_k: int = 10, version_id: str | None = None,
                  params: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.time()
    rec, vid, p = get_recommender("similar_item", version_id, params)
    if house_id not in rec.index_of:
        raise KeyError(f"房源不存在: {house_id}")
    i = rec.index_of[house_id]
    anchor = {
        "house_id": house_id,
        "district": str(rec.df.at[i, "district_name"]),
        "area": float(rec.df.at[i, "area"]),
        "unit_price": float(rec.df.at[i, "unit_price"]),
        "total_price": float(rec.df.at[i, "total_price"]),
        "rooms": int(rec.df.at[i, "room_total"]) if "room_total" in rec.df.columns else None,
        "build_year": int(rec.df.at[i, "build_year"]) if pd.notna(rec.df.at[i, "build_year"]) else None,
        "renovation": str(rec.df.at[i, "renovation_name"]) if "renovation_name" in rec.df.columns else "",
        "floor": str(rec.df.at[i, "floor_category"]) if "floor_category" in rec.df.columns else "",
    }
    items = rec.recommend_by_id(house_id, top_k=top_k)
    return {"algo_id": "similar_item", "version_id": vid, "params": p,
            "anchor": anchor, "count": len(items), "items": items,
            "seconds": round(time.time() - t0, 3)}


__all__ = [
    "train_price", "predict_price", "recommend", "similar_items",
    "get_recommender", "load_version_data", "load_split", "list_models",
    "get_model_record", "delete_model", "clear_caches", "MODEL_REGISTRY",
]
