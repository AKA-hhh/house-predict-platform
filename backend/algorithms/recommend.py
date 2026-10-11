"""
推荐算法模块（Recommenders）
=============================
实现五个可切换、可配置的推荐器，覆盖任务要求的全部方法家族：

  1. popularity      热度基线（可解释，设定效果下限）
  2. content_tfidf   内容推荐（TF-IDF + 余弦相似度）
  3. item_cf         协同过滤（Item-based 余弦 / SVD 矩阵分解）
  4. hybrid          混合推荐（多路召回 + 加权排序 + 硬约束过滤）
  5. similar_item    相似房源检索（特征空间近邻，可在**真实数据**上评估）

⚠️ 数据限制与处理方式（重要）
  本公开数据集**没有用户 ID 字段**，无法直接构造真实「用户-房源」交互矩阵。
  因此协同过滤部分采用**参数化行为模拟日志**：用户画像（偏好区域、面积区间、
  价格区间、户型）从真实数据分布中采样，再按画像采样交互项，并保留随机探索项。
  该模拟过程**完全由随机种子决定、参数公开、可复现**，是学术界在缺少用户日志时
  构造离线评估集的常规做法。真实数据可评估的部分（相似房源检索）另设独立协议。
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .metrics import ndcg_at_k, precision_at_k, ranking_report, recall_at_k
from .registry import get_algorithm, validate_params

RANDOM_STATE = 42
MAX_VOCAB = 1200


# ============================================================ 工具函数
def _norm01(a: np.ndarray) -> np.ndarray:
    """min-max 归一化到 [0,1]，全等时返回全 0。"""
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _item_uid(df: pd.DataFrame) -> np.ndarray:
    if "house_id" in df.columns:
        return df["house_id"].astype(str).values
    return np.arange(len(df)).astype(str)


def item_text_series(df: pd.DataFrame) -> pd.Series:
    """把结构化字段拼成房源文本画像，供 TF-IDF 使用。"""
    def col(name: str, default: str = "未知") -> pd.Series:
        if name in df.columns:
            return df[name].astype(str).fillna(default)
        return pd.Series([default] * len(df), index=df.index)

    age = pd.Series(["未知"] * len(df), index=df.index)
    if "building_age" in df.columns:
        age = pd.cut(df["building_age"], bins=[-1, 5, 15, 30, 999],
                     labels=["次新", "较新", "较旧", "老旧"]).astype(str)

    parts = [
        "区" + col("district_name"),
        col("building_type_name").map(lambda x: "型" + x),
        col("renovation_name").map(lambda x: "装" + x),
        col("floor_category").map(lambda x: "层" + x),
        col("elevator_name").map(lambda x: "梯" + x),
        col("subway_name").map(lambda x: "铁" + x),
        col("five_years_name").map(lambda x: "满" + x),
        col("area_bucket").map(lambda x: "面" + x + "平"),
        col("room_total").map(lambda x: f"室{int(float(x)) if str(x).replace('.','').isdigit() else 0}"),
        age.map(lambda x: "龄" + x),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out


def profile_to_text(profile: dict[str, Any]) -> str:
    """用户偏好 -> 文本查询（与 item_text_series 使用同一套词表前缀）。"""
    toks: list[str] = []
    if profile.get("district"):
        toks.append("区" + str(profile["district"]))
    if profile.get("building_type"):
        toks.append("型" + str(profile["building_type"]))
    if profile.get("renovation"):
        toks.append("装" + str(profile["renovation"]))
    if profile.get("floor_category"):
        toks.append("层" + str(profile["floor_category"]))
    if profile.get("elevator_required"):
        toks.append("梯有电梯")
    if profile.get("subway_required"):
        toks.append("铁近地铁")
    if profile.get("five_years_required"):
        toks.append("满满五年")
    if profile.get("area_bucket"):
        toks.append("面" + str(profile["area_bucket"]) + "平")
    if profile.get("rooms"):
        toks.append(f"室{int(profile['rooms'])}")
    return " ".join(toks) if toks else "区朝阳区"


# ============================================================ 模拟交互日志
def build_interaction_log(
    df: pd.DataFrame,
    n_users: int = 800,
    n_interactions: int = 12,
    seed: int = 42,
    explore_rate: float = 0.2,
) -> dict[str, Any]:
    """构造隐式反馈日志（用户画像采样 + 行为采样）。

    返回 {log: DataFrame[user_id, house_id, weight], profiles: [dict], stats: {...}}
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    uid = _item_uid(df)
    unit = pd.to_numeric(df.get("unit_price"), errors="coerce").fillna(0).values
    area = pd.to_numeric(df.get("area"), errors="coerce").fillna(0).values
    dist = (df["district_name"].astype(str).values
            if "district_name" in df.columns else np.array(["未知"] * len(df)))

    uniq, counts = np.unique(dist, return_counts=True)
    p = counts / counts.sum()

    # 预建区域 -> 索引池
    pools: dict[str, np.ndarray] = {}
    for d in uniq:
        pools[d] = np.where(dist == d)[0]

    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []

    for u in range(n_users):
        d = str(rng.choice(uniq, p=p))
        pool = pools[d]
        if len(pool) < 30:
            pool = np.arange(len(df))
        sub_price = unit[pool]
        med = float(np.median(sub_price)) if len(sub_price) else 40000.0
        ratio = rng.uniform(0.55, 1.6)
        lo, hi = med * (ratio - 0.35), med * (ratio + 0.35)
        area_lo, area_hi = sorted(rng.uniform(35, 180, 2))

        cand = pool[
            (unit[pool] >= lo) & (unit[pool] <= hi) &
            (area[pool] >= area_lo) & (area[pool] <= area_hi)
        ]
        if len(cand) < n_interactions:
            cand = pool
        n_exp = max(1, int(n_interactions * explore_rate))
        n_exploit = max(1, n_interactions - n_exp)
        picked = list(rng.choice(cand, size=min(n_exploit, len(cand)), replace=False))
        picked += list(rng.choice(len(df), size=n_exp, replace=False))

        for idx in picked:
            rows.append({
                "user_id": f"U{u:04d}",
                "house_id": uid[int(idx)],
                "weight": float(rng.uniform(0.5, 1.0)),
            })
        profiles.append({
            "user_id": f"U{u:04d}",
            "district": d,
            "price_lo": round(lo, 0), "price_hi": round(hi, 0),
            "area_lo": round(area_lo, 1), "area_hi": round(area_hi, 1),
            "ratio": round(ratio, 3),
        })

    log = pd.DataFrame(rows).drop_duplicates(subset=["user_id", "house_id"])
    return {
        "log": log,
        "profiles": profiles,
        "stats": {
            "n_users": int(log["user_id"].nunique()),
            "n_items": int(log["house_id"].nunique()),
            "n_interactions": int(len(log)),
            "sparsity": round(1 - len(log) / max(log["user_id"].nunique() * len(df), 1), 6),
            "seed": seed, "explore_rate": explore_rate,
            "seconds": round(time.time() - t0, 2),
            "note": "用户行为模拟日志（公开数据集无真实用户 ID），协议与种子公开可复现",
        },
    }


# ============================================================ 基类
class BaseRecommender:
    algo_id: str = "base"
    name: str = "推荐器"

    def fit(self, df: pd.DataFrame, **kwargs) -> "BaseRecommender":
        self.df = df.reset_index(drop=True)
        self.uids = _item_uid(self.df)
        self.index_of = {u: i for i, u in enumerate(self.uids)}
        return self

    def score_all(self, user, top_k=10, exclude=None) -> np.ndarray:
        """返回对**全量候选**的打分向量（长度 = 商品数）。子类必须实现。"""
        raise NotImplementedError

    def recommend(self, user: dict[str, Any], top_k: int = 10,
                  exclude: Iterable[str] | None = None) -> list[dict[str, Any]]:
        k = int(user.get("top_k") or top_k)
        scores = self.score_all(user, top_k=k, exclude=exclude)
        idx = np.argsort(-scores)[:k]
        return self._pack(idx, scores, len(idx), lambda i: self._reason_default(i))

    def _reason_default(self, i: int) -> list[str]:
        return []

    def recommend_by_id(self, house_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ----------------------------------------------------- 结果包装
    def _pack(self, idx: np.ndarray, scores: np.ndarray, k: int,
              reasons_fn=None) -> list[dict[str, Any]]:
        out = []
        for i in np.asarray(idx)[:k]:
            i = int(i)
            rec = {
                "house_id": str(self.uids[i]),
                "score": round(float(scores[i]), 4) if scores is not None else None,
                "district": str(self.df.at[i, "district_name"]) if "district_name" in self.df.columns else "",
                "area": round(float(self.df.at[i, "area"]), 1) if "area" in self.df.columns else None,
                "unit_price": round(float(self.df.at[i, "unit_price"]), 0) if "unit_price" in self.df.columns else None,
                "total_price": round(float(self.df.at[i, "total_price"]), 1) if "total_price" in self.df.columns else None,
                "rooms": int(self.df.at[i, "room_total"]) if "room_total" in self.df.columns else None,
                "floor": str(self.df.at[i, "floor_category"]) if "floor_category" in self.df.columns else "",
                "build_year": int(self.df.at[i, "build_year"]) if "build_year" in self.df.columns and pd.notna(self.df.at[i, "build_year"]) else None,
                "renovation": str(self.df.at[i, "renovation_name"]) if "renovation_name" in self.df.columns else "",
                "elevator": str(self.df.at[i, "elevator_name"]) if "elevator_name" in self.df.columns else "",
                "subway": str(self.df.at[i, "subway_name"]) if "subway_name" in self.df.columns else "",
                "followers": int(self.df.at[i, "followers"]) if "followers" in self.df.columns and pd.notna(self.df.at[i, "followers"]) else None,
            }
            if reasons_fn is not None:
                rec["reasons"] = reasons_fn(i)
            out.append(rec)
        return out


# ============================================================ 1. 热度基线
class PopularityRecommender(BaseRecommender):
    algo_id = "popularity"
    name = "热度基线"

    def fit(self, df, **kwargs):
        super().fit(df)
        p = validate_params("popularity", kwargs.get("params") or {})
        self.params = p
        n = len(self.df)
        followers = pd.to_numeric(self.df.get("followers", pd.Series(np.zeros(n))), errors="coerce").fillna(0).values
        follow_score = _norm01(np.log1p(followers))
        dom = pd.to_numeric(self.df.get("days_on_market", pd.Series(np.zeros(n))), errors="coerce").fillna(0).values
        fresh_score = 1 - _norm01(dom)
        if "district_name" in self.df.columns and "unit_price" in self.df.columns:
            med = self.df.groupby("district_name")["unit_price"].transform("median")
            value = ((med - self.df["unit_price"]) / med.replace(0, np.nan)).fillna(0).clip(-1, 1)
            value_score = _norm01(value.values)
        else:
            value_score = np.zeros(n)
        self.scores_ = (p["w_followers"] * follow_score
                        + p["w_freshness"] * fresh_score
                        + p["w_value"] * value_score)
        self.parts_ = {"follow": follow_score, "fresh": fresh_score, "value": value_score}
        self.order_ = np.argsort(-self.scores_)
        for k, v in self.parts_.items():
            self.df[f"_pop_{k}"] = v
        return self

    def _score_all(self) -> np.ndarray:
        return self.scores_

    def score_all(self, user, top_k=10, exclude=None) -> np.ndarray:
        scores = self.scores_.copy()
        if exclude:
            for h in exclude:
                if h in self.index_of:
                    scores[self.index_of[h]] = -1e9
        if user.get("district") and "district_name" in self.df.columns:
            scores = np.where(self.df["district_name"].values == user["district"], scores, scores * 0.85)
        return scores

    def recommend(self, user, top_k=10, exclude=None):
        k = int(user.get("top_k") or top_k)
        scores = self.score_all(user, top_k=k, exclude=exclude)
        idx = np.argsort(-scores)[:k]
        return self._pack(idx, scores, len(idx), self._reason)

    def _reason(self, i: int) -> list[str]:
        r = [f"关注度分位 {self.df.at[i, '_pop_follow']:.2f}", f"成交新鲜度 {self.df.at[i, '_pop_fresh']:.2f}"]
        if self.df.at[i, "_pop_value"] > 0.6:
            r.append("单价低于同区中位数（性价比高）")
        return r

    def recommend_by_id(self, house_id, top_k=10):
        if house_id not in self.index_of:
            return []
        i = self.index_of[house_id]
        d = self.df.at[i, "district_name"] if "district_name" in self.df.columns else None
        user = {"district": d, "top_k": top_k}
        return self.recommend(user, top_k, exclude=[house_id])


# ============================================================ 2. 内容推荐
class ContentRecommender(BaseRecommender):
    algo_id = "content_tfidf"
    name = "内容推荐（TF-IDF）"

    def fit(self, df, **kwargs):
        super().fit(df)
        p = validate_params("content_tfidf", kwargs.get("params") or {})
        self.params = p
        docs = item_text_series(self.df)
        self.vec = TfidfVectorizer(
            tokenizer=str.split, lowercase=False, token_pattern=None,
            ngram_range=(1, p["ngram_max"]), min_df=p["min_df"], max_features=MAX_VOCAB,
        )
        self.T = self.vec.fit_transform(docs)

        # 数值相似度分量
        num_cols = [c for c in ("area", "unit_price", "build_year", "room_total", "dist_to_center")
                    if c in self.df.columns]
        if num_cols:
            num = self.df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
            self.scaler = StandardScaler().fit(num)
            self.N = self.scaler.transform(num)
            self.N = self.N / (np.linalg.norm(self.N, axis=1, keepdims=True) + 1e-9)
        else:
            self.N = None
        self.num_cols = num_cols
        return self

    def _scores_for_query(self, query_text: str) -> np.ndarray:
        q = self.vec.transform([query_text])
        if q.nnz == 0:                      # 查询词全部 OOV：退化为全 0，由调用方回退
            return np.zeros(len(self.df))
        sim = (self.T @ q.T).toarray().ravel()
        sim = sim / (np.linalg.norm(q.toarray()) + 1e-9)
        return sim

    def _scores_for_item(self, i: int) -> np.ndarray:
        sim = (self.T @ self.T[i].T).toarray().ravel()
        if self.N is not None:
            nsim = self.N @ self.N[i]
            sim = (1 - self.params["numeric_weight"]) * sim + self.params["numeric_weight"] * nsim
        return sim

    def score_all(self, user, top_k=10, exclude=None) -> np.ndarray:
        text = user.get("query_text") or profile_to_text(user)
        sim = self._scores_for_query(text)
        nsim = np.zeros(len(self.df))
        if self.N is not None and user.get("_anchor_index") is not None:
            nsim = self.N @ self.N[int(user["_anchor_index"])]
        w = self.params["numeric_weight"]
        scores = (1 - w) * sim + w * nsim if self.N is not None else sim
        # 文本全 OOV 时退化为纯数值/热度，避免全 0 造成"无推荐"
        if sim.max() <= 0 and self.N is None:
            scores = np.zeros(len(self.df))
        scores = _hard_filter(self.df, scores, user, self.params.get("hard_filter", False))
        if exclude:
            for h in exclude:
                if h in self.index_of:
                    scores[self.index_of[h]] = -1e9
        return scores

    def recommend(self, user, top_k=10, exclude=None):
        text = user.get("query_text") or profile_to_text(user)
        k = int(user.get("top_k") or top_k)
        scores = self.score_all(user, top_k=k, exclude=exclude)
        idx = np.argsort(-scores)[:k]
        return self._pack(idx, scores, len(idx), lambda i: self._reason(i, text))

    def _reason(self, i: int, text: str) -> list[str]:
        doc = item_text_series(self.df.iloc[[i]]).iloc[0]
        toks = set(text.split())
        hit = [t for t in doc.split() if t in toks]
        return [f"命中偏好：{', '.join(hit[:5])}"] if hit else ["内容相似度较高"]

    def recommend_by_id(self, house_id, top_k=10):
        if house_id not in self.index_of:
            return []
        i = self.index_of[house_id]
        scores = self._scores_for_item(i)
        scores[i] = -1e9
        idx = np.argsort(-scores)[:top_k]
        return self._pack(idx, scores, len(idx), lambda j: ["同小区/同类型、面积与单价接近"])


# ============================================================ 3. 协同过滤
class ItemCFRecommender(BaseRecommender):
    algo_id = "item_cf"
    name = "协同过滤（Item-based）"

    def fit(self, df, **kwargs):
        super().fit(df)
        p = validate_params("item_cf", kwargs.get("params") or {})
        self.params = p
        log = kwargs.get("log")
        if log is None or len(log) == 0:
            log = build_interaction_log(df, p["n_users"], p["n_interactions"], p["seed"])["log"]
        self.log = log
        self.log_stats = kwargs.get("log_stats") or {"n_interactions": int(len(log))}

        users = sorted(log["user_id"].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        item_ids = self.uids
        self.item_index = self.index_of

        r = log["user_id"].map(self.user_index).values
        c = log["house_id"].map(self.item_index).values
        mask = ~pd.isna(c)
        r, c = r[mask].astype(int), c[mask].astype(int)
        w = log["weight"].values[mask] if "weight" in log.columns else np.ones(len(r))
        M = sp.csr_matrix((w, (c, r)), shape=(len(item_ids), len(users)), dtype=np.float32)

        if p["method"] == "svd":
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=min(p["n_factors"], max(2, min(M.shape) - 1)),
                                    random_state=p["seed"])
            self.U = self.svd.fit_transform(M.T)             # (n_users, k)
            self.V = self.svd.components_.T                   # (n_items, k)
            self.M = M
            self.Mn = None
        else:
            norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
            inv = 1.0 / (norms + p["shrinkage"])
            D = sp.diags(inv)
            self.Mn = (D @ M).tocsr()                          # 列归一化后的 item 向量
            self.M = M
            self.svd = None
            self.U = self.V = None

        self.user_items = log.groupby("user_id")["house_id"].agg(list).to_dict()
        return self

    def _scores_for_user(self, user_id: str) -> np.ndarray:
        items = [self.index_of[h] for h in self.user_items.get(user_id, []) if h in self.index_of]
        if not items:
            return np.zeros(len(self.df))
        if self.params["method"] == "svd":
            v = self.U[self.user_index[user_id]]
            return self.V @ v
        v = np.asarray(self.Mn[items].sum(axis=0)).ravel()
        return np.asarray(self.Mn @ v).ravel()

    def score_all(self, user, top_k=10, exclude=None) -> np.ndarray:
        uid = user.get("user_id")
        if uid and uid in self.user_index:
            scores = self._scores_for_user(uid)
        else:
            scores = np.zeros(len(self.df))
        if self.params.get("hard_filter"):
            scores = _hard_filter(self.df, scores, user, True)
        if exclude:
            for h in exclude:
                if h in self.index_of:
                    scores[self.index_of[h]] = -1e9
        return scores

    def recommend(self, user, top_k=10, exclude=None):
        k = int(user.get("top_k") or top_k)
        scores = self.score_all(user, top_k=k, exclude=exclude)
        idx = np.argsort(-scores)[:k]
        return self._pack(idx, scores, len(idx), self._reason)

    def _reason(self, i: int) -> list[str]:
        return ["与你交互过的房源相似的买家也关注了它"]

    def recommend_by_id(self, house_id, top_k=10):
        if house_id not in self.index_of or self.Mn is None:
            return []
        i = self.index_of[house_id]
        sim = np.asarray((self.Mn @ self.Mn[i].T).todense()).ravel()
        sim[i] = -1e9
        idx = np.argsort(-sim)[:top_k]
        return self._pack(idx, sim, len(idx), lambda j: ["在隐式反馈中共现相似"])


# ============================================================ 4. 相似房源
class SimilarItemRecommender(BaseRecommender):
    algo_id = "similar_item"
    name = "相似房源检索"

    def fit(self, df, **kwargs):
        super().fit(df)
        p = validate_params("similar_item", kwargs.get("params") or {})
        self.params = p
        feats = self.df.copy()
        num_cols = [c for c in ("area", "unit_price", "room_total", "build_year",
                                "dist_to_center", "floor_number", "ladder_ratio")
                    if c in feats.columns]
        cat_cols = [c for c in ("district_name", "building_type_name", "renovation_name",
                                "floor_category", "elevator_name", "subway_name")
                    if c in feats.columns]
        num = feats[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.scaler = StandardScaler().fit(num.values)
        Xn = self.scaler.transform(num.values)
        if cat_cols:
            dummies = pd.get_dummies(feats[cat_cols].astype(str), prefix=cat_cols).astype(float).values
            Xn = np.hstack([Xn, dummies * 2.0])       # 类别特征加权，与链家"同区位同类型"直觉一致
        self.X = Xn
        if p["metric"] == "cosine":
            self.Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9)
        else:
            self.Xn = Xn
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        return self

    def _scores(self, i: int) -> np.ndarray:
        if self.params["metric"] == "cosine":
            return self.Xn @ self.Xn[i]
        return -np.linalg.norm(self.Xn - self.Xn[i], axis=1)

    def recommend_by_id(self, house_id, top_k=10):
        if house_id not in self.index_of:
            return []
        i = self.index_of[house_id]
        scores = self._scores(i).copy()
        if self.params["same_district"] and "district_name" in self.df.columns:
            same = self.df["district_name"].values == self.df.at[i, "district_name"]
            scores = np.where(same, scores, -1e9)
        if "area" in self.df.columns:
            a = self.df["area"].values
            tol = self.params["area_tol"]
            ok = (np.abs(a - a[i]) <= tol * max(a[i], 1))
            scores = np.where(ok, scores, scores - 100)
        scores[i] = -1e9
        idx = np.argsort(-scores)[:top_k]
        return self._pack(idx, scores, len(idx), lambda j: self._diff(i, j))

    def _diff(self, i: int, j: int) -> list[str]:
        diffs = []
        for c in self.num_cols:
            vi, vj = self.df.at[i, c], self.df.at[j, c]
            if pd.notna(vi) and pd.notna(vj):
                diffs.append(f"{c}: {round(float(vi), 1)} → {round(float(vj), 1)}")
        return diffs[:3]

    def score_all(self, user, top_k=10, exclude=None) -> np.ndarray:
        d = user.get("district")
        if d and "district_name" in self.df.columns:
            pool = np.where(self.df["district_name"].values == d)[0]
        else:
            pool = np.arange(len(self.df))
        if len(pool) == 0:
            pool = np.arange(len(self.df))
        anchor = int(pool[np.argmin(np.abs(self.df["area"].values[pool] - float(user.get("area") or 90)))])
        self._anchor = anchor
        scores = self._scores(anchor).copy()
        scores[anchor] = -1e9
        if exclude:
            for h in exclude:
                if h in self.index_of:
                    scores[self.index_of[h]] = -1e9
        return scores

    def recommend(self, user, top_k=10, exclude=None):
        k = int(user.get("top_k") or top_k)
        scores = self.score_all(user, top_k=k, exclude=exclude)
        ref = getattr(self, "_anchor", 0)
        idx = np.argsort(-scores)[:k]
        return self._pack(idx, scores, len(idx), lambda j: self._diff(ref, j))


# ============================================================ 5. 混合推荐
class HybridRecommender(BaseRecommender):
    algo_id = "hybrid"
    name = "混合推荐（召回 + 排序）"

    def fit(self, df, **kwargs):
        super().fit(df)
        p = validate_params("hybrid", kwargs.get("params") or {})
        self.params = p
        log = kwargs.get("log")
        self.stages: dict[str, Any] = {}
        self.content = ContentRecommender().fit(df, params={"numeric_weight": 0.35})
        self.pop = PopularityRecommender().fit(df, params={})
        self.cf = None
        if p["w_cf"] > 0:
            try:
                self.cf = ItemCFRecommender().fit(df, params={
                    "method": "item_cosine", "shrinkage": 8.0, "n_users": 500, "n_interactions": 12,
                }, log=log)
            except Exception as exc:  # CF 失败时自动降级为双路混合
                self.stages["cf_error"] = str(exc)
        self.stages["recall_size"] = p["recall_size"]
        self.stages["routes"] = ["content"] + (["cf"] if self.cf else []) + ["popularity"]
        return self

    def score_all(self, user, top_k=10, exclude=None) -> np.ndarray:
        n = len(self.df)
        s_pop = self.pop.score_all(user)
        if self.params["hard_filter"]:
            s_pop = _hard_filter(self.df, s_pop, user, True)

        # 召回层（内容 + 协同过滤 + 热度），各自归一化
        routes: dict[str, np.ndarray] = {"pop": s_pop}
        profile_for_content = {k: v for k, v in user.items() if k != "user_id"}
        s_content = self.content.score_all(profile_for_content)
        if self.content._scores_for_query(profile_for_content.get("query_text")
                                          or profile_to_text(profile_for_content)).max() > 0:
            routes["content"] = s_content
        if self.cf is not None and user.get("user_id") in getattr(self.cf, "user_index", {}):
            routes["cf"] = self.cf.score_all(user)

        parts, weights = [], []
        for name, arr in routes.items():
            parts.append(_norm01(arr))
            weights.append({"content": self.params["w_content"],
                            "cf": self.params["w_cf"],
                            "pop": self.params["w_pop"]}[name])
        wsum = sum(weights) or 1.0
        final = sum(w / wsum * arr for w, arr in zip(weights, parts))

        # 硬约束过滤（内容与 CF 两路）
        if self.params["hard_filter"]:
            mask = _constraint_mask(self.df, user)
            if mask is not None:
                final = np.where(mask, final, -1e9)

        if exclude:
            for h in exclude:
                if h in self.index_of:
                    final[self.index_of[h]] = -1e9
        self._routes = routes
        return final

    def recommend(self, user, top_k=10, exclude=None):
        k = int(user.get("top_k") or top_k)
        final = self.score_all(user, top_k=k, exclude=exclude)
        idx = np.argsort(-final)[:k]
        nc = _norm01(self._routes.get("content", np.zeros(len(self.df))))
        nf = _norm01(self._routes.get("cf", np.zeros(len(self.df))))
        npp = _norm01(self._routes.get("pop", np.zeros(len(self.df))))
        return self._pack(idx, final, len(idx), lambda i: self._reason(i, nc, nf, npp))

    def _reason(self, i: int, nc, nf, npp) -> list[str]:
        parts = []
        if nc[i] > 0.5:
            parts.append(f"内容偏好匹配度 {nc[i]:.2f}")
        if nf[i] > 0.3:
            parts.append(f"协同过滤分 {nf[i]:.2f}")
        if npp[i] > 0.6:
            parts.append(f"市场热度分 {npp[i]:.2f}")
        return parts or ["综合得分靠前"]

    def recommend_by_id(self, house_id, top_k=10):
        recs = self.content.recommend_by_id(house_id, top_k)
        return recs


# ============================================================ 硬约束
def _constraint_mask(df: pd.DataFrame, user: dict[str, Any]) -> np.ndarray | None:
    """返回布尔掩码：True 表示满足用户硬约束。无任何约束时返回 None。"""
    mask = np.ones(len(df), dtype=bool)
    used = False
    if user.get("budget_max") and "total_price" in df.columns:
        mask &= df["total_price"].values <= float(user["budget_max"]) * 1.05; used = True
    if user.get("budget_min") and "total_price" in df.columns:
        mask &= df["total_price"].values >= float(user["budget_min"]) * 0.95; used = True
    if user.get("area_min") and "area" in df.columns:
        mask &= df["area"].values >= float(user["area_min"]) * 0.9; used = True
    if user.get("area_max") and "area" in df.columns:
        mask &= df["area"].values <= float(user["area_max"]) * 1.1; used = True
    if user.get("rooms") and "room_total" in df.columns:
        mask &= df["room_total"].values >= float(user["rooms"]); used = True
    if user.get("subway_required") and "subway" in df.columns:
        mask &= df["subway"].values == 1; used = True
    if user.get("elevator_required") and "elevator" in df.columns:
        mask &= df["elevator"].values == 1; used = True
    if user.get("district") and "district_name" in df.columns:
        mask &= df["district_name"].values == user["district"]; used = True
    return mask if used else None


def _hard_filter(df: pd.DataFrame, scores: np.ndarray, user: dict[str, Any],
                 enabled: bool) -> np.ndarray:
    if not enabled:
        return scores
    mask = _constraint_mask(df, user)
    if mask is None:
        return scores
    return np.where(mask, scores, -1e9)


# ============================================================ 工厂
def build_recommender(algo_id: str, df: pd.DataFrame, params: dict[str, Any] | None = None,
                      log: pd.DataFrame | None = None, log_stats: dict | None = None):
    cls = {
        "popularity": PopularityRecommender,
        "content_tfidf": ContentRecommender,
        "item_cf": ItemCFRecommender,
        "hybrid": HybridRecommender,
        "similar_item": SimilarItemRecommender,
    }.get(algo_id)
    if cls is None:
        raise KeyError(f"未注册的推荐算法: {algo_id}")
    return cls().fit(df, params=params or {}, log=log, log_stats=log_stats)


# ============================================================ 离线评估
def profile_from_items(df: pd.DataFrame, items: Sequence[str]) -> dict[str, Any]:
    """由用户交互过的房源集合，反推一个可用的偏好画像（用于内容/混合推荐）。"""
    sub = df[df["house_id"].astype(str).isin({str(i) for i in items})]
    prof: dict[str, Any] = {}
    if not len(sub):
        return prof
    cat_keys = [("district_name", "district"), ("building_type_name", "building_type"),
                ("renovation_name", "renovation"), ("floor_category", "floor_category")]
    for col, key in cat_keys:
        if col in sub.columns:
            m = sub[col].mode(dropna=True)
            if len(m):
                prof[key] = m.iloc[0]
    for col, key in (("elevator", "elevator_required"), ("subway", "subway_required"),
                     ("five_years", "five_years_required")):
        if col in sub.columns:
            prof[key] = bool(pd.to_numeric(sub[col], errors="coerce").fillna(0).mean() >= 0.5)
    for col, key in (("area", "area"), ("room_total", "rooms")):
        if col in sub.columns:
            v = pd.to_numeric(sub[col], errors="coerce").median()
            if pd.notna(v):
                prof[key] = float(v)
    if "area_bucket" in sub.columns:
        m = sub["area_bucket"].mode(dropna=True)
        if len(m):
            prof["area_bucket"] = m.iloc[0]
    return prof


def evaluate_recommenders(
    df: pd.DataFrame,
    algo_ids: Sequence[str] = ("popularity", "content_tfidf", "item_cf", "hybrid"),
    params_map: dict[str, dict] | None = None,
    ks: Sequence[int] = (5, 10, 20),
    n_users: int = 800,
    n_interactions: int = 12,
    seed: int = 42,
    max_eval_users: int = 200,
    n_negatives: int = 999,
) -> dict[str, Any]:
    """基于留一法（leave-one-out）的推荐器离线对比。

    协议：
      1. 每个用户的最后一条交互作为测试正例，其余为训练交互；
      2. **采样候选集排序**：正例 + 从全体房源中随机采样的 `n_negatives` 条负例，
         在该子集内排序并计算指标。

    ⚠️ 为什么用采样候选集而不是全量排序：
       候选房源有 31.8 万条，而每个用户只有 1 条测试正例。若在全量目录上排序，
       随机命中概率约为 1/318200，所有算法的指标都会被稀释到 0 附近而失去区分度。
       「1 正例 + N 随机负例」是推荐系统离线评估的通行做法
       （如 NCF 论文使用 100 个负例、GRU4Rec 的采样评估），
       本项目采用 N=999 以降低随机性，并在结果中明确标注该协议。
    """
    t0 = time.time()
    params_map = params_map or {}
    log_pack = build_interaction_log(df, n_users, n_interactions, seed)
    log = log_pack["log"].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    tail = log.groupby("user_id").tail(1)
    held = tail.set_index("user_id")["house_id"].to_dict()
    train_log = log.drop(tail.index).reset_index(drop=True)

    users = list(held.keys())
    rng = np.random.default_rng(seed)
    eval_users = list(rng.choice(users, size=min(max_eval_users, len(users)), replace=False))

    uid = _item_uid(df)
    uid_to_idx = {u: i for i, u in enumerate(uid)}
    n_items = len(df)

    # 预生成负例候选（每个评测用户一组，互不重复以降低抽样偏差）
    neg_idx = rng.choice(n_items, size=(len(eval_users), n_negatives), replace=False)

    train_by_user = train_log.groupby("user_id")["house_id"].agg(list).to_dict()

    reports: dict[str, Any] = {}
    for algo_id in algo_ids:
        t_algo = time.time()
        try:
            rec = build_recommender(algo_id, df, params_map.get(algo_id), log=train_log,
                                    log_stats=log_pack["stats"])
        except Exception as exc:  # noqa: BLE001
            reports[algo_id] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        per_recs: dict[str, list] = {}
        for ui, u in enumerate(eval_users):
            profile = profile_from_items(df, train_by_user.get(u, []))
            profile["user_id"] = u
            profile["top_k"] = max(ks)
            try:
                scores = rec.score_all(profile)
            except Exception:  # noqa: BLE001  单个算法失败不影响整体
                scores = np.zeros(n_items)

            pos = uid_to_idx.get(str(held[u]))
            cand = list(neg_idx[ui])
            if pos is not None:
                cand = [pos] + cand
            cand = np.asarray(cand, dtype=int)
            order = np.argsort(-scores[cand])
            per_recs[u] = [str(uid[cand[j]]) for j in order[:max(ks)]]

        per_rel = {u: {str(held[u])} for u in eval_users}
        rep = ranking_report(per_recs, per_rel, ks=ks, n_items=n_items)
        rep["seconds"] = round(time.time() - t_algo, 2)
        rep["algo_id"] = algo_id
        rep["algo_name"] = get_algorithm(algo_id)["name"]
        reports[algo_id] = rep

    return {
        "protocol": {
            "type": "leave-one-out + sampled-candidate ranking",
            "n_users_in_log": len(users),
            "n_eval_users": len(eval_users),
            "n_interactions_per_user": n_interactions,
            "n_candidates_per_user": n_negatives + 1,
            "catalog_size": int(n_items),
            "seed": seed,
            "caveat": "交互日志为参数化模拟（公开数据集无用户 ID），指标用于算法间相对比较，不代表线上绝对值；"
                      "候选集为 1 正例 + 随机负例的采样协议，指标数值会高于全量目录排序。",
        },
        "reports": reports,
        "log_stats": log_pack["stats"],
        "seconds": round(time.time() - t0, 2),
    }


def evaluate_similarity_on_real_data(
    df: pd.DataFrame,
    k_list: Sequence[int] = (5, 10, 20),
    n_queries: int = 300,
    seed: int = 42,
    price_tol: float = 0.10,
) -> dict[str, Any]:
    """**真实数据**上的相似房源检索评估。

    相关性定义：与查询房源同小区 且 单价相对差 ≤ price_tol。
    该定义只依赖真实成交记录，不引入任何模拟数据。
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    rec = SimilarItemRecommender().fit(df, params={"metric": "cosine", "same_district": True,
                                                   "area_tol": 0.5})
    n = len(df)
    q_idx = rng.choice(n, size=min(n_queries, n), replace=False)

    uid = rec.uids
    unit = pd.to_numeric(df["unit_price"], errors="coerce").values
    cid = df["community_id"].values if "community_id" in df.columns else None

    hits = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}
    for i in q_idx:
        candidates = np.arange(n)
        if cid is not None:
            rel_mask = (cid == cid[i]) & (np.abs(unit - unit[i]) / max(unit[i], 1) <= price_tol)
        else:
            rel_mask = np.abs(unit - unit[i]) / max(unit[i], 1) <= price_tol
        rel_mask[i] = False
        relevant = set(uid[rel_mask])
        if not relevant:
            continue
        scores = rec._scores(int(i))
        scores[int(i)] = -1e9
        order = np.argsort(-scores)[:max(k_list)]
        recs = [str(uid[j]) for j in order]
        for k in k_list:
            hits[k].append(precision_at_k(recs, relevant, k))
            ndcgs[k].append(ndcg_at_k(recs, relevant, k))

    report = {
        "protocol": "real-data similarity retrieval",
        "relevance_definition": f"同小区(community_id) 且 |单价相对差| ≤ {price_tol:.0%}",
        "n_queries_used": len(hits[k_list[0]]),
        "n_queries_sampled": int(len(q_idx)),
        "note": "仅使用真实成交记录，无模拟数据；用于验证相似度检索的可用性",
        "seconds": round(time.time() - t0, 2),
    }
    for k in k_list:
        if hits[k]:
            report[f"Precision@{k}"] = round(float(np.mean(hits[k])), 4)
            report[f"NDCG@{k}"] = round(float(np.mean(ndcgs[k])), 4)
    report["avg_relevant_per_query"] = round(float(np.mean([
        len(set(uid[(cid == cid[i]) & (np.abs(unit - unit[i]) / max(unit[i], 1) <= price_tol)]))
        for i in q_idx[:50]
    ])) if cid is not None and len(q_idx) else 0.0, 1)
    return report


__all__ = [
    "build_interaction_log", "item_text_series", "profile_to_text", "profile_from_items",
    "build_recommender", "evaluate_recommenders", "evaluate_similarity_on_real_data",
    "PopularityRecommender", "ContentRecommender", "ItemCFRecommender",
    "HybridRecommender", "SimilarItemRecommender",
]
