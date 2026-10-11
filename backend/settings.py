"""
全局配置与路径管理
==================
集中定义项目目录、数据库位置、默认参数，供各模块复用。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 目录结构
BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REGISTRY_DIR = DATA_DIR / "registry"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"
EXPERIMENT_DIR = BASE_DIR / "experiments" / "results"
WEB_DIR = BACKEND_DIR / "web"

DB_PATH = DATA_DIR / "house.db"
SOURCE_CONFIG_PATH = CONFIG_DIR / "datasources.json"

# ---------------------------------------------------------------- 运行参数
# 分块读取大小：体现"批处理 / 分区扫描"思路，避免一次性载入超大文件
DEFAULT_CHUNK_SIZE = 200_000
# 候选编码，按顺序探测
ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1")


def ensure_dirs() -> None:
    """确保所有运行期目录存在（幂等）。"""
    for d in (
        CONFIG_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR,
        REGISTRY_DIR, MODEL_DIR, REPORT_DIR, EXPERIMENT_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


ensure_dirs()
