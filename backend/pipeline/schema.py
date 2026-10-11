"""
数据模式（Schema）定义
======================
定义原始链家二手房数据集的**列映射**与**码值字典**。

原始数据来源：
    链家网（lianjia.com）北京二手房成交记录，2011-11 ~ 2017-02
    原始文件字段为中文业务语义的拼音/英文缩写混合命名，且类别字段使用数字编码，
    因此需要集中的映射表把「原始列」翻译为「规范列」，并把「编码值」翻译为「可读标签」。
"""
from __future__ import annotations

# ------------------------------------------------------------------ 列映射
# 原始列名 -> 规范列名
COLUMN_MAP: dict[str, str] = {
    "url": "url",
    "id": "house_id",
    "Lng": "lng",
    "Lat": "lat",
    "Cid": "community_id",
    "tradeTime": "trade_time",
    "DOM": "days_on_market",          # Days On Market，挂牌到成交天数
    "followers": "followers",          # 关注人数（隐式反馈代理指标）
    "totalPrice": "total_price",       # 总价，单位：万元
    "price": "unit_price",             # 单价，单位：元/平米  <-- 主目标变量
    "square": "area",                  # 建筑面积，单位：平米
    "livingRoom": "living_room",
    "drawingRoom": "drawing_room",
    "kitchen": "kitchen",
    "bathRoom": "bath_room",
    "floor": "floor_raw",              # 形如 "高 26" / "中 12" / "低 3"
    "buildingType": "building_type",
    "constructionTime": "build_year",
    "renovationCondition": "renovation",
    "buildingStructure": "building_structure",
    "ladderRatio": "ladder_ratio",     # 梯户比
    "elevator": "elevator",
    "fiveYearsProperty": "five_years",
    "subway": "subway",
    "district": "district_code",
    "communityAverage": "community_avg_price",
}

# 需要保留的规范列（顺序即最终输出顺序）
CANONICAL_COLUMNS: list[str] = [
    "house_id", "url", "trade_time", "district_code", "district_name",
    "community_id", "community_avg_price",
    "lng", "lat",
    "total_price", "unit_price", "area",
    "living_room", "drawing_room", "kitchen", "bath_room", "room_total",
    "floor_raw", "floor_category", "floor_number", "floor_ratio",
    "build_year", "building_age", "building_type", "building_structure",
    "renovation", "ladder_ratio", "elevator", "five_years", "subway",
    "days_on_market", "followers",
]

# ------------------------------------------------------------- 码值字典
# 注：链家原始数据集的类别字段为匿名数字编码，以下映射依据该数据集公开说明
#     与常见解读整理；无法确证的编码保留为 "未知(编码N)"，不做臆测。
DISTRICT_MAP: dict[int, str] = {
    1: "东城区", 2: "丰台区", 3: "通州区", 4: "大兴区", 5: "房山区",
    6: "昌平区", 7: "朝阳区", 8: "海淀区", 9: "石景山区", 10: "西城区",
    11: "平谷区", 12: "门头沟区", 13: "密云区",
}

BUILDING_TYPE_MAP: dict[int, str] = {
    1: "塔楼", 2: "板楼", 3: "板塔结合", 4: "平房",
}

BUILDING_STRUCTURE_MAP: dict[int, str] = {
    1: "未知结构", 2: "混合结构", 3: "砖木结构",
    4: "砖混结构", 5: "钢混结构", 6: "钢结构",
}

RENOVATION_MAP: dict[int, str] = {
    1: "毛坯", 2: "简装", 3: "精装", 4: "豪华装修",
}

FLOOR_CATEGORY_MAP: dict[str, str] = {
    "低": "低楼层", "中": "中楼层", "高": "高楼层",
}

# 数值列（用于类型强制与异常检测）
NUMERIC_COLUMNS: list[str] = [
    "lng", "lat", "community_id", "community_avg_price",
    "total_price", "unit_price", "area",
    "living_room", "drawing_room", "kitchen", "bath_room",
    "build_year", "ladder_ratio", "elevator", "five_years", "subway",
    "days_on_market", "followers", "district_code",
]

# 业务合理区间（用于异常值识别，单位同规范列）
VALID_RANGES: dict[str, tuple[float, float]] = {
    "unit_price": (3_000.0, 200_000.0),      # 元/平米
    "total_price": (20.0, 15_000.0),         # 万元
    "area": (10.0, 1_000.0),                 # 平米
    "build_year": (1900.0, 2020.0),
    "ladder_ratio": (0.0, 1.0),
    "days_on_market": (0.0, 4_000.0),
    "community_avg_price": (3_000.0, 250_000.0),
}

# 目标变量的业务硬区间：越界的记录直接剔除（而非截断），因为它们多半是数据错误。
# 注意：统计方法（IQR / Z-Score）对右偏价格分布只能识别"过高"的异常，
#      低端异常因为下界为负而漏检（详见 docs/05 问题清单 A11），因此需要业务区间兜底。
TARGET_BUSINESS_RANGE = (3_000.0, 200_000.0)

# 需要做异常值处理的业务列。
#
# ⚠️ 有意**排除目标变量 `unit_price`**：对标签做统计截断等于"改标签"而不是"清洗数据"——
#    本项目实测 IQR 上界为 92,477 元/㎡，会把最高的 3.7%（11,751 条）真实高价成交
#    全部压成同一个值，使模型在该区间被要求预测一个被人为改写的目标（详见 docs/05 A15）。
#    目标变量的异常处置改为**业务区间硬过滤**（TARGET_BUSINESS_RANGE，剔除而非改写）。
#
# 有意**不包含** days_on_market / followers / kitchen / bath_room 等重尾计数型变量：
# 这类变量用 IQR 截断会把整列压成中位数附近，破坏信息（详见 docs/05 A11）。
OUTLIER_COLUMNS: list[str] = [
    "total_price", "area", "community_avg_price", "build_year", "ladder_ratio",
]

# 异常值处置动作
OUTLIER_ACTIONS: list[str] = ["winsorize", "clip_median", "drop"]


TARGET_COLUMN = "unit_price"

# 用于建模的数值特征（去除标识、时间戳原文与目标本身）
MODEL_NUMERIC_FEATURES: list[str] = [
    "area", "living_room", "drawing_room", "kitchen", "bath_room", "room_total",
    "floor_category_code", "floor_number", "floor_ratio",
    "build_year", "building_age", "building_type_code", "building_structure_code",
    "renovation_code", "ladder_ratio", "elevator", "five_years", "subway",
    "days_on_market", "followers", "community_avg_price",
    "lng", "lat", "trade_year", "trade_month", "trade_quarter",
]

# 用于推荐/相似度计算的分类型与数值混合特征
RECOMMEND_FEATURES: list[str] = [
    "district_name", "building_type_name", "renovation_name",
    "floor_category", "elevator_name", "subway_name", "five_years_name",
    "area", "unit_price", "room_total", "build_year",
]

# 特征集档位（profile）：供训练时切换，用于「特征消融」这一迭代优化环节。
#   full     : 默认全量特征
#   no_time  : 去掉成交时间分量。原因：时间切分下，树模型无法外推未见过的年份，
#              分年特征会让模型把周期模式记死而无法泛化（详见 E3 消融实验）。
#   compact  : 只保留房屋自身属性，去掉区位与小区信息，作为"最少信息"对照。
TIME_FEATURES: list[str] = ["trade_year", "trade_month", "trade_quarter"]
GEO_FEATURES: list[str] = ["lng", "lat", "dist_to_center", "is_core_district"]
COMMUNITY_FEATURES: list[str] = ["community_avg_price"]

FEATURE_PROFILES: dict[str, dict[str, Any]] = {
    "full": {"name": "全量特征", "drop": []},
    "no_time": {"name": "精简（去除成交时间分量）", "drop": TIME_FEATURES},
    "no_community": {"name": "去除小区均价", "drop": COMMUNITY_FEATURES},
    "compact": {"name": "仅房屋属性（去区位与小区）",
                "drop": TIME_FEATURES + GEO_FEATURES + COMMUNITY_FEATURES},
}


def resolve_feature_profile(profile: str) -> list[str]:
    """把特征档位名解析为具体特征列表。"""
    p = FEATURE_PROFILES.get(profile or "full", FEATURE_PROFILES["full"])
    drop = set(p["drop"])
    return [f for f in MODEL_NUMERIC_FEATURES if f not in drop]


# 中文标签列（由编码翻译而来，供前端展示与文本特征使用）
LABEL_COLUMNS: list[str] = [
    "district_name", "building_type_name", "building_structure_name",
    "renovation_name", "elevator_name", "five_years_name", "subway_name",
]
