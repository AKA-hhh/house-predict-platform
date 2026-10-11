"""
算法注册表（Algorithm Registry）
=================================
系统里每一个可选算法的**自描述元数据**都集中在这里，前端据此自动渲染"算法选择 + 参数配置"表单，
后端据此校验参数，实验脚本据此做批量对比。新增算法只需在此登记 + 在实现模块中注册构造函数。

每个算法声明：
  - 任务类型 task        : price_regression | recommendation
  - 是否可解释 interpretable
  - 输入 / 输出契约
  - 主要参数（类型、默认值、取值范围、说明）
  - 评价方法 metrics
  - 适用场景与已知局限
"""
from __future__ import annotations

from typing import Any

ALGORITHMS: dict[str, dict[str, Any]] = {
    # ================================================== 价格预测
    "linear_ols": {
        "id": "linear_ols",
        "name": "线性回归基线（OLS）",
        "task": "price_regression",
        "category": "线性模型",
        "interpretable": True,
        "summary": "最小二乘线性回归。系数可直接解释为「该特征每变动 1 个单位，单价变动多少元/㎡」，"
                   "是可解释性最强的基线，用于给后续复杂模型设定性能下限。",
        "inputs": {
            "X": "数值特征矩阵 (n_samples × n_features)：area、room_total、building_age、"
                 "floor_ratio、dist_to_center、community_avg_price、经纬度、成交时间分量等",
            "y": "目标向量 unit_price（元/㎡）",
        },
        "outputs": {
            "coef": "每个特征的回归系数 + 截距（可解释性来源）",
            "metrics": "MAE / RMSE / R² / MAPE / 误差带命中率",
        },
        "params": [
            {"name": "fit_intercept", "type": "bool", "default": True, "desc": "是否拟合截距"},
            {"name": "positive", "type": "bool", "default": False, "desc": "强制系数为正（部分场景更符合业务直觉）"},
        ],
        "metrics": ["MAE", "RMSE", "R2", "MAPE(%)", "Within10%(%)", "Within20%(%)"],
        "pros": ["完全可解释", "训练极快", "系数方向可作为业务结论"],
        "cons": ["无法刻画非线性与特征交互", "对异常值与共线性敏感"],
    },
    "ridge": {
        "id": "ridge",
        "name": "岭回归（L2 正则线性）",
        "task": "price_regression",
        "category": "线性模型",
        "interpretable": True,
        "summary": "在 OLS 上加入 L2 正则，缓解经纬度、面积等强相关特征导致的共线性问题，"
                   "系数依然可解释，通常比 OLS 更稳。",
        "inputs": {"X": "同线性基线（建议先标准化）", "y": "unit_price"},
        "outputs": {"coef": "正则化后的标准化系数", "metrics": "同上"},
        "params": [
            {"name": "alpha", "type": "float", "default": 1.0, "min": 0.0, "max": 1000.0, "desc": "L2 正则强度，越大越保守"},
        ],
        "metrics": ["MAE", "RMSE", "R2", "MAPE(%)", "Within10%(%)"],
        "pros": ["可解释", "抗共线性", "几乎无调参成本"],
        "cons": ["仍为线性假设"],
    },
    "hgb": {
        "id": "hgb",
        "name": "梯度提升树（HistGradientBoosting）",
        "task": "price_regression",
        "category": "树模型",
        "interpretable": False,
        "summary": "直方图分箱的梯度提升树，能自动捕捉非线性与特征交互，在本数据集上通常显著优于线性模型；"
                   "可解释性通过「置换重要性 + 单样本偏离度归因」近似提供。",
        "inputs": {"X": "数值特征矩阵（支持缺失值）", "y": "unit_price"},
        "outputs": {"prediction": "预测单价", "importance": "置换重要性排序", "metrics": "全套回归指标"},
        "params": [
            {"name": "learning_rate", "type": "float", "default": 0.08, "min": 0.01, "max": 1.0, "desc": "学习率"},
            {"name": "max_iter", "type": "int", "default": 300, "min": 20, "max": 2000, "desc": "提升轮数"},
            {"name": "max_depth", "type": "int", "default": 8, "min": 2, "max": 30, "desc": "单树最大深度（None 表示不限）"},
            {"name": "min_samples_leaf", "type": "int", "default": 40, "min": 1, "max": 500, "desc": "叶节点最小样本数"},
            {"name": "l2_regularization", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0, "desc": "L2 正则"},
            {"name": "max_bins", "type": "int", "default": 255, "min": 32, "max": 255, "desc": "直方图分箱数"},
        ],
        "metrics": ["MAE", "RMSE", "R2", "MAPE(%)", "Within10%(%)", "Within20%(%)"],
        "pros": ["精度高", "自动非线性/交互", "训练速度快于同精度 XGBoost"],
        "cons": ["黑盒，需要额外解释手段", "参数较多"],
    },
    "rf": {
        "id": "rf",
        "name": "随机森林",
        "task": "price_regression",
        "category": "树模型",
        "interpretable": False,
        "summary": "Bagging 集成，方差小、抗过拟合，提供原生的特征重要性，作为与梯度提升的对照。",
        "inputs": {"X": "数值特征矩阵", "y": "unit_price"},
        "outputs": {"prediction": "预测单价", "importance": "Gini 重要性", "metrics": "全套回归指标"},
        "params": [
            {"name": "n_estimators", "type": "int", "default": 120, "min": 10, "max": 800, "desc": "树数量"},
            {"name": "max_depth", "type": "int", "default": 16, "min": 2, "max": 60, "desc": "最大深度"},
            {"name": "min_samples_leaf", "type": "int", "default": 5, "min": 1, "max": 200, "desc": "叶节点最小样本"},
            {"name": "max_features", "type": "str", "default": "sqrt", "options": ["sqrt", "log2", "1.0"], "desc": "分裂时考虑的特征比例"},
            {"name": "n_jobs", "type": "int", "default": -1, "desc": "并行进程数"},
        ],
        "metrics": ["MAE", "RMSE", "R2", "MAPE(%)"],
        "pros": ["稳健", "重要性直观"],
        "cons": ["内存/时间开销大", "外推能力差"],
    },

    # ================================================== 推荐
    "popularity": {
        "id": "popularity",
        "name": "热度基线（可解释基线）",
        "task": "recommendation",
        "category": "基线",
        "interpretable": True,
        "summary": "按「关注人数 × 新鲜度 × 单价性价比」的加权分数排序推荐。规则完全透明，"
                   "推荐理由可以直接给出（如「近地铁 / 关注度高 / 单价低于同区中位数 12%」），"
                   "同时为协同过滤与内容推荐设定效果下限。",
        "inputs": {"df": "房源明细（含 followers / days_on_market / trade_time / unit_price / district_name）"},
        "outputs": {"items": "Top-K 房源 ID 列表", "scores": "每个房源的加权分数", "reasons": "可读推荐理由"},
        "params": [
            {"name": "w_followers", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "desc": "关注度权重"},
            {"name": "w_freshness", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "desc": "成交新鲜度权重"},
            {"name": "w_value", "type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "desc": "性价比（低于同区均价比例）权重"},
            {"name": "top_k", "type": "int", "default": 10, "min": 1, "max": 50, "desc": "返回条数"},
        ],
        "metrics": ["Precision@K", "Recall@K", "NDCG@K", "HitRate@K"],
        "pros": ["无需用户数据", "理由可解释", "冷启动友好"],
        "cons": ["个性化程度低", "易造成热门聚集"],
    },
    "content_tfidf": {
        "id": "content_tfidf",
        "name": "内容推荐（TF-IDF + 余弦相似度）",
        "task": "recommendation",
        "category": "内容/相似度",
        "interpretable": True,
        "summary": "把房源的类别特征（区域、建筑类型、装修、楼层、电梯、地铁、满五年）拼成文本，"
                   "用 TF-IDF 向量化；把用户偏好同样转为文本向量，用余弦相似度召回。"
                   "对每个推荐结果可给出命中/未命中的偏好项作为解释。",
        "inputs": {
            "items_text": "房源文本画像（结构化字段拼接）",
            "user_profile": "{区域, 户型, 面积区间, 预算, 装修, 必须近地铁, 必须电梯, 楼层偏好}",
        },
        "outputs": {"items": "Top-K 房源 ID", "scores": "余弦相似度", "reasons": "偏好命中说明"},
        "params": [
            {"name": "ngram_max", "type": "int", "default": 2, "min": 1, "max": 3, "desc": "TF-IDF n-gram 上限"},
            {"name": "min_df", "type": "int", "default": 2, "min": 1, "max": 50, "desc": "词频最小文档数"},
            {"name": "numeric_weight", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "desc": "数值特征在相似度中的权重"},
            {"name": "hard_filter", "type": "bool", "default": False, "desc": "是否对用户显式约束（预算/面积/户型/区域/地铁/电梯）做硬过滤。离线评估默认关闭以保持算法纯净；界面提交真实偏好时由服务层自动开启"},
            {"name": "top_k", "type": "int", "default": 10, "min": 1, "max": 50, "desc": "返回条数"},
        ],
        "metrics": ["Precision@K", "Recall@K", "NDCG@K", "HitRate@K", "Coverage"],
        "pros": ["无需用户历史", "可解释", "冷启动好"],
        "cons": ["依赖人工特征", "难以发现跨品类惊喜项"],
    },
    "item_cf": {
        "id": "item_cf",
        "name": "协同过滤（Item-based CF / 余弦相似度）",
        "task": "recommendation",
        "category": "协同过滤",
        "interpretable": True,
        "summary": "在「用户-房源」隐式反馈矩阵上计算**房源-房源余弦相似度**，"
                   "再按用户历史交互过的房源聚合打分召回。另有 SVD 矩阵分解模式作为对照。",
        "inputs": {
            "interaction": "隐式反馈矩阵（用户 × 房源 0/1 或加权）",
            "note": "本公开数据集**不含用户 ID**，因此采用带随机种子的、参数化的行为模拟日志"
                    "（用户画像从真实数据分布采样）构造反馈矩阵，属标准离线评估做法；"
                    "协议与种子完全公开，可复现。",
        },
        "outputs": {"items": "Top-K 房源 ID", "scores": "聚合相似度/预测评分", "neighbors": "支撑该推荐的相似房源"},
        "params": [
            {"name": "method", "type": "str", "default": "item_cosine", "options": ["item_cosine", "svd"], "desc": "相似度计算 / 矩阵分解"},
            {"name": "n_users", "type": "int", "default": 800, "min": 50, "max": 5000, "desc": "模拟用户数"},
            {"name": "n_interactions", "type": "int", "default": 12, "min": 3, "max": 60, "desc": "每用户交互条数"},
            {"name": "n_factors", "type": "int", "default": 32, "min": 4, "max": 256, "desc": "SVD 隐向量维度"},
            {"name": "shrinkage", "type": "float", "default": 10.0, "min": 0.0, "max": 100.0, "desc": "相似度收缩系数（抑制低共现噪声）"},
            {"name": "top_k", "type": "int", "default": 10, "min": 1, "max": 50, "desc": "返回条数"},
            {"name": "seed", "type": "int", "default": 42, "desc": "随机种子"},
            {"name": "hard_filter", "type": "bool", "default": False, "desc": "是否对用户显式约束做硬过滤。离线评估默认关闭以保持算法纯净；界面提交真实偏好时由服务层自动开启"},
        ],
        "metrics": ["Precision@K", "Recall@K", "NDCG@K", "HitRate@K", "Coverage"],
        "pros": ["能挖掘非显式偏好", "不依赖内容特征"],
        "cons": ["冷启动差", "本数据集缺真实用户 ID，需模拟日志"],
    },
    "hybrid": {
        "id": "hybrid",
        "name": "混合推荐（多路召回 + 加权排序）",
        "task": "recommendation",
        "category": "混合/召回排序",
        "interpretable": True,
        "summary": "召回层并行跑「内容推荐 / Item-CF / 热度」三路，合并去重后再用加权公式排序："
                   "final = w1·归一化内容分 + w2·归一化CF分 + w3·归一化热度分，"
                   "并叠加「必须满足」的硬约束过滤（预算上限、户型、区域）。",
        "inputs": {"user_profile": "同内容推荐", "interaction": "同 Item-CF（可选）"},
        "outputs": {"items": "Top-K", "scores": "融合分数", "reasons": "来自哪几路召回 + 命中约束"},
        "params": [
            {"name": "w_content", "type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "desc": "内容路权重"},
            {"name": "w_cf", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "desc": "协同过滤路权重"},
            {"name": "w_pop", "type": "float", "default": 0.20, "min": 0.0, "max": 1.0, "desc": "热度路权重"},
            {"name": "recall_size", "type": "int", "default": 200, "min": 20, "max": 2000, "desc": "每路召回条数"},
            {"name": "hard_filter", "type": "bool", "default": True, "desc": "是否启用预算/户型/区域硬约束"},
            {"name": "top_k", "type": "int", "default": 10, "min": 1, "max": 50, "desc": "返回条数"},
        ],
        "metrics": ["Precision@K", "Recall@K", "NDCG@K", "HitRate@K", "Coverage"],
        "pros": ["兼顾个性化与冷启动", "鲁棒", "可解释来源"],
        "cons": ["权重需要调参", "实现复杂度高"],
    },
    "similar_item": {
        "id": "similar_item",
        "name": "相似房源检索（特征空间近邻）",
        "task": "recommendation",
        "category": "相似度计算",
        "interpretable": True,
        "summary": "对给定房源，在标准化后的数值+类别特征空间中检索最相似的 K 条房源"
                   "（余弦/欧氏距离 + 区域/户型硬过滤），即链家「相似房源」的算法等价物。"
                   "该模块的评估可在**真实数据**上以「同小区且单价差 ≤10%」为相关性定义做离线评价。",
        "inputs": {"query_item": "房源 ID + 其完整特征", "metric": "cosine | euclidean"},
        "outputs": {"items": "Top-K 相似房源 ID", "distances": "距离/相似度", "reasons": "差异最大的维度"},
        "params": [
            {"name": "metric", "type": "str", "default": "cosine", "options": ["cosine", "euclidean"], "desc": "距离度量"},
            {"name": "same_district", "type": "bool", "default": True, "desc": "是否限定同区域"},
            {"name": "area_tol", "type": "float", "default": 0.3, "min": 0.05, "max": 2.0, "desc": "面积允许偏差比例"},
            {"name": "top_k", "type": "int", "default": 10, "min": 1, "max": 50, "desc": "返回条数"},
        ],
        "metrics": ["Precision@K", "NDCG@K", "平均特征差异度"],
        "pros": ["纯真实数据即可评估", "解释直观", "可复用于房源详情页"],
        "cons": ["非个性化", "高维稀疏下需谨慎选距离度量"],
    },
}

TASKS: dict[str, dict[str, Any]] = {
    "price_regression": {
        "id": "price_regression",
        "name": "二手房单价预测",
        "target": "unit_price（元/㎡）",
        "default_algorithm": "linear_ols",
        "metrics": ["MAE", "RMSE", "R2", "MAPE(%)", "Within10%(%)", "Within20%(%)"],
    },
    "recommendation": {
        "id": "recommendation",
        "name": "房源推荐",
        "target": "Top-K 房源列表",
        "default_algorithm": "hybrid",
        "metrics": ["Precision@K", "Recall@K", "NDCG@K", "HitRate@K", "Coverage"],
    },
}


def list_algorithms(task: str | None = None) -> list[dict[str, Any]]:
    items = list(ALGORITHMS.values())
    if task:
        items = [a for a in items if a["task"] == task]
    return items


def get_algorithm(algo_id: str) -> dict[str, Any]:
    if algo_id not in ALGORITHMS:
        raise KeyError(f"未注册的算法: {algo_id}")
    return ALGORITHMS[algo_id]


def default_params(algo_id: str) -> dict[str, Any]:
    return {p["name"]: p["default"] for p in get_algorithm(algo_id)["params"]}


def validate_params(algo_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """按注册表校验并补全参数，返回清洗后的参数字典。"""
    spec = get_algorithm(algo_id)
    out: dict[str, Any] = {}
    for p in spec["params"]:
        name = p["name"]
        val = params.get(name, p["default"])
        t = p["type"]
        try:
            if t == "int":
                val = int(val)
            elif t == "float":
                val = float(val)
            elif t == "bool":
                val = bool(val) if not isinstance(val, str) else val.lower() in ("1", "true", "yes")
            else:
                val = str(val)
        except (TypeError, ValueError):
            val = p["default"]
        if "min" in p and isinstance(val, (int, float)):
            val = max(val, p["min"])
        if "max" in p and isinstance(val, (int, float)):
            val = min(val, p["max"])
        if "options" in p and val not in p["options"]:
            val = p["default"]
        out[name] = val
    return out


__all__ = [
    "ALGORITHMS", "TASKS", "list_algorithms", "get_algorithm",
    "default_params", "validate_params",
]
