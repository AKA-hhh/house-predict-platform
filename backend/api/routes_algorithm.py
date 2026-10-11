"""
模块二 · 核心算法 API（算法注册表 / 模型训练 / 价格预测 / 房源推荐 / 离线评估）
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from .. import jobs
from ..algorithms import registry as algo_reg
from ..algorithms.trainer import (
    clear_caches, delete_model, get_model_record, list_models, predict_price,
    recommend, similar_items, train_price,
)
from ..pipeline.versioning import versions
from ..settings import EXPERIMENT_DIR

router = APIRouter(prefix="/api", tags=["核心算法"])


# ------------------------------------------------------------ 算法注册表
@router.get("/algorithms")
def list_algorithms(task: str = "") -> dict[str, Any]:
    from ..pipeline.schema import FEATURE_PROFILES
    return {
        "tasks": list(algo_reg.TASKS.values()),
        "items": algo_reg.list_algorithms(task or None),
        "feature_profiles": [
            {"id": k, "name": v["name"], "drop": v["drop"]} for k, v in FEATURE_PROFILES.items()
        ],
    }


@router.get("/algorithms/{algo_id}")
def get_algorithm(algo_id: str) -> dict[str, Any]:
    try:
        spec = algo_reg.get_algorithm(algo_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    spec = dict(spec)
    spec["defaults"] = algo_reg.default_params(algo_id)
    return spec


# ------------------------------------------------------------ 价格模型
@router.post("/models/train")
def train(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    algo_id = payload.get("algo_id")
    if not algo_id:
        raise HTTPException(status_code=400, detail="缺少 algo_id")
    spec = algo_reg.get_algorithm(algo_id)
    if spec["task"] != "price_regression":
        raise HTTPException(status_code=400, detail=f"{algo_id} 不是价格预测算法")
    params = payload.get("params") or {}
    version_id = payload.get("version_id") or None
    features = payload.get("features") or None
    target_mode = payload.get("target_mode") or "absolute"
    feature_profile = payload.get("feature_profile") or "full"

    if payload.get("async", True):
        job_id = jobs.submit("train", train_price, algo_id, params, version_id, features,
                             payload.get("tags", ""), target_mode, feature_profile)
        return {"job_id": job_id, "mode": "async"}
    return {"mode": "sync", "result": train_price(algo_id, params, version_id, features,
                                                  payload.get("tags", ""), target_mode,
                                                  feature_profile)}


@router.get("/models")
def models(task: str = "") -> dict[str, Any]:
    items = list_models(task or None)
    return {"items": items, "best": _best_by_rmse(items)}


def _best_by_rmse(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    cand = [m for m in items if (m.get("metrics") or {}).get("test")]
    if not cand:
        return None
    best = min(cand, key=lambda m: m["metrics"]["test"].get("RMSE", 1e18))
    return {"model_id": best["model_id"], "algo_name": best["algo_name"],
            "RMSE": best["metrics"]["test"].get("RMSE"),
            "R2": best["metrics"]["test"].get("R2")}


@router.get("/models/{model_id}")
def model_detail(model_id: str) -> dict[str, Any]:
    try:
        rec = get_model_record(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    rec = dict(rec)
    # 附带该模型的训练结果明细（若实验记录中存在）
    p = EXPERIMENT_DIR / f"train_{model_id}.json"
    if p.exists():
        try:
            rec["detail"] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return rec


@router.delete("/models/{model_id}")
def remove_model(model_id: str) -> dict[str, Any]:
    delete_model(model_id)
    clear_caches()
    return {"ok": True}


@router.post("/models/{model_id}/predict")
def do_predict(model_id: str, inputs: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return predict_price(model_id, inputs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predict")
def predict_auto(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """未指定模型时，自动选用当前数据版本下 R² 最优的价格模型。"""
    model_id = payload.get("model_id")
    if not model_id:
        items = list_models("price_regression")
        cand = [m for m in items if (m.get("metrics") or {}).get("test")]
        if not cand:
            raise HTTPException(status_code=400, detail="尚无已训练的价格模型，请先训练")
        model_id = max(cand, key=lambda m: m["metrics"]["test"].get("R2", -1e18))["model_id"]
    return predict_price(model_id, payload.get("inputs") or {})


# ------------------------------------------------------------ 推荐
@router.post("/recommend")
def do_recommend(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    algo_id = payload.get("algo_id") or "hybrid"
    profile = payload.get("profile") or {}
    try:
        return recommend(
            algo_id, profile,
            top_k=int(payload.get("top_k") or 10),
            version_id=payload.get("version_id") or None,
            params=payload.get("params") or None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.post("/recommend/similar")
def do_similar(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    house_id = payload.get("house_id")
    if not house_id:
        raise HTTPException(status_code=400, detail="缺少 house_id")
    try:
        return similar_items(
            str(house_id),
            top_k=int(payload.get("top_k") or 10),
            version_id=payload.get("version_id") or None,
            params=payload.get("params") or None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.get("/recommend/sample-id")
def sample_house_id(n: int = 1) -> dict[str, Any]:
    """随机给出若干房源 ID，方便前端"试一试"。"""
    from ..algorithms.trainer import load_version_data
    v = versions.active_version()
    if not v:
        raise HTTPException(status_code=400, detail="尚无可用的数据版本")
    df = load_version_data(v["version_id"])
    s = df.sample(min(n, len(df)), random_state=None)
    return {"items": [
        {"house_id": str(r["house_id"]), "district": str(r.get("district_name")),
         "area": float(r["area"]), "unit_price": float(r["unit_price"])}
        for _, r in s.iterrows()
    ]}


@router.post("/recommend/evaluate")
def evaluate(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    algo_ids = payload.get("algo_ids") or ["popularity", "content_tfidf", "item_cf", "hybrid"]
    version_id = payload.get("version_id") or None
    max_users = int(payload.get("max_eval_users") or 200)
    n_users = int(payload.get("n_users") or 800)
    jobs_id = jobs.submit(
        "evaluate_recommend", _run_recommend_eval,
        algo_ids, version_id, max_users, n_users,
    )
    return {"job_id": jobs_id, "mode": "async"}


def _run_recommend_eval(algo_ids: list, version_id: str | None, max_users: int,
                        n_users: int) -> dict[str, Any]:
    from ..algorithms.recommend import evaluate_recommenders, evaluate_similarity_on_real_data
    from ..algorithms.trainer import load_version_data
    v = versions.active_version() if version_id is None else versions.get(version_id)
    if v is None:
        raise RuntimeError("尚无可用的数据版本")
    df = load_version_data(v["version_id"])
    rep = evaluate_recommenders(df, algo_ids, ks=(5, 10, 20), n_users=n_users,
                                max_eval_users=max_users)
    rep["similarity_real_data"] = evaluate_similarity_on_real_data(df, n_queries=200)
    rep["version_id"] = v["version_id"]
    out = EXPERIMENT_DIR / "recommend_eval.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


__all__ = ["router"]
