"""
数据源配置管理
==============
把"数据从哪来"从代码里解耦出来：所有数据源以 JSON 配置持久化，
前端「数据源配置」页面与独立导入工具共享同一份配置。

支持的 source_type：
  - local_csv : 本地 CSV 文件（推荐，离线可复现）
  - url_csv   : 远程 CSV（HTTP/HTTPS 直链，支持断点续传下载到 raw 目录）
  - sqlite    : 已导入的 SQLite 表（用于增量更新 / 二次处理）
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..settings import SOURCE_CONFIG_PATH, RAW_DIR, DATA_DIR, BASE_DIR


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class DataSource:
    source_id: str
    name: str
    source_type: str                     # local_csv | url_csv | sqlite
    uri: str                             # 文件路径 / URL / sqlite 表名
    encoding: str = "auto"               # auto 表示自动探测
    delimiter: str = ","
    description: str = ""
    city: str = "北京"
    license: str = ""
    attribution: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    last_run_at: str = ""
    last_run_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------ 默认数据源
DEFAULT_SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "lianjia_bj_raw",
        "name": "链家·北京二手房成交（原始）",
        "source_type": "local_csv",
        "uri": str(RAW_DIR / "new.csv"),
        "encoding": "gb18030",
        "delimiter": ",",
        "city": "北京",
        "description": "链家网北京二手房成交记录 2011-11 ~ 2017-02，约 31.8 万条，26 个字段。",
        "license": "CC0-1.0 (Public Domain)",
        "attribution": (
            "Lianjia.com 公开成交数据；整理版见 GitHub: "
            "Logical-Dhruv/housing-price-in-beijing (new.csv)；"
            "同源清洗版见 Kaggle: dhruvconnects/beijing-housing-price-cleaned"
        ),
        "options": {"header": True, "chunk_size": 200000},
    },
    {
        "source_id": "lianjia_bj_cleaned",
        "name": "链家·北京二手房成交（上游清洗版）",
        "source_type": "url_csv",
        "uri": "https://raw.githubusercontent.com/Logical-Dhruv/housing-price-in-beijing/main/cleaned_data.csv",
        "encoding": "utf-8",
        "delimiter": ",",
        "city": "北京",
        "description": "上游仓库已做过一次清洗/字段翻译的数据，可用于对照实验。",
        "license": "CC0-1.0",
        "attribution": "Logical-Dhruv/housing-price-in-beijing",
        "options": {"header": True, "chunk_size": 200000},
    },
]


class SourceRegistry:
    """数据源注册表：JSON 文件持久化，进程内缓存。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else SOURCE_CONFIG_PATH
        self._cache: dict[str, DataSource] | None = None

    # ---------------------------------------------------------- 读写
    def _load_raw(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {"sources": []}
        return {"sources": []}

    def _bootstrap(self) -> dict[str, DataSource]:
        raw = self._load_raw()
        items = raw.get("sources") or []
        if not items:
            items = [dict(s) for s in DEFAULT_SOURCES]
            self._write(items)
        store: dict[str, DataSource] = {}
        for it in items:
            known = {k: v for k, v in it.items() if k in DataSource.__dataclass_fields__}
            store[known["source_id"]] = DataSource(**known)
        return store

    def _write(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": _now(), "sources": items}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _flush(self) -> None:
        assert self._cache is not None
        self._write([s.to_dict() for s in self._cache.values()])

    @property
    def sources(self) -> dict[str, DataSource]:
        if self._cache is None:
            self._cache = self._bootstrap()
        return self._cache

    # ---------------------------------------------------------- CRUD
    def list(self) -> list[dict[str, Any]]:
        out = []
        for s in self.sources.values():
            d = s.to_dict()
            d["exists"] = self.exists(s)
            d["resolved_path"] = self.resolved_path(s)
            out.append(d)
        return out

    def get(self, source_id: str) -> DataSource:
        if source_id not in self.sources:
            raise KeyError(f"数据源不存在: {source_id}")
        return self.sources[source_id]

    def upsert(self, payload: dict[str, Any]) -> DataSource:
        sid = payload.get("source_id") or f"src_{uuid.uuid4().hex[:8]}"
        if sid in self.sources:
            cur = self.sources[sid]
            for k, v in payload.items():
                if k in DataSource.__dataclass_fields__ and k != "source_id":
                    setattr(cur, k, v)
            cur.updated_at = _now()
            cur_source = cur
        else:
            known = {k: v for k, v in payload.items() if k in DataSource.__dataclass_fields__}
            known["source_id"] = sid
            cur_source = DataSource(**known)
            self.sources[sid] = cur_source
        self._flush()
        return cur_source

    def delete(self, source_id: str) -> None:
        self.sources.pop(source_id, None)
        self._flush()

    def mark_run(self, source_id: str, rows: int) -> None:
        s = self.get(source_id)
        s.last_run_at = _now()
        s.last_run_rows = rows
        self._flush()

    # ------------------------------------------------------- 路径解析
    @staticmethod
    def resolved_path(src: DataSource) -> str:
        if src.source_type == "local_csv":
            p = Path(src.uri)
            if not p.is_absolute():
                # 相对路径优先相对项目根目录解析，其次相对 data/ 目录
                cand = BASE_DIR / p
                p = cand if cand.exists() else (DATA_DIR / p)
            return str(p)
        if src.source_type == "url_csv":
            return str(RAW_DIR / Path(src.uri).name)
        return src.uri

    def exists(self, src: DataSource) -> bool:
        if src.source_type == "sqlite":
            from ..settings import DB_PATH
            return DB_PATH.exists()
        return Path(self.resolved_path(src)).exists()


registry = SourceRegistry()
