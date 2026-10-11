"""
数据版本管理（Data Versioning）
================================
每一次流水线运行都会登记为一个**数据版本**，记录：
  - 来源数据源与原始文件指纹（SHA-256）
  - 各处理阶段（导入/清洗/特征/划分）的**参数、输入行数、输出行数、耗时**
  - 最终产物文件（Parquet / SQLite 表）路径与校验和
  - 列级统计指纹（均值/缺失率等），用于快速判断两份数据是否等价

设计目标：可复现、可对比、可回溯。任何一次建模都能追溯到具体的数据版本。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..settings import REGISTRY_DIR


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    """流式计算文件 SHA-256，避免整文件读入内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(block)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def fingerprint_dataframe(df: pd.DataFrame, max_cols: int = 40) -> dict[str, Any]:
    """对 DataFrame 生成轻量统计指纹：行数、列数、各列缺失率与分位数摘要。"""
    fp: dict[str, Any] = {"rows": int(len(df)), "cols": int(df.shape[1]), "columns": {}}
    for col in list(df.columns)[:max_cols]:
        s = df[col]
        info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_rate": round(float(s.isna().mean()), 6),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            q = s.quantile([0.25, 0.5, 0.75])
            info.update({
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4) if s.notna().sum() > 1 else 0.0,
                "min": round(float(s.min()), 4),
                "q25": round(float(q.loc[0.25]), 4),
                "median": round(float(q.loc[0.5]), 4),
                "q75": round(float(q.loc[0.75]), 4),
                "max": round(float(s.max()), 4),
            })
        fp["columns"][col] = info
    return fp


class VersionRegistry:
    """数据版本登记簿：一个版本一个 JSON 文件 + 一份全局 index.json。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else REGISTRY_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    # --------------------------------------------------------- index
    def load_index(self) -> list[dict[str, Any]]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_index(self, items: list[dict[str, Any]]) -> None:
        self.index_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def new_version_id(self, prefix: str = "ds") -> str:
        return f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"

    # -------------------------------------------------------- 写版本
    def save(self, record: dict[str, Any]) -> dict[str, Any]:
        vid = record["version_id"]
        (self.root / f"{vid}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        idx = self.load_index()
        idx = [i for i in idx if i.get("version_id") != vid]
        idx.insert(0, {
            "version_id": vid,
            "created_at": record.get("created_at"),
            "source_id": record.get("source_id"),
            "source_name": record.get("source_name"),
            "status": record.get("status"),
            "raw_rows": record.get("raw_rows"),
            "final_rows": record.get("final_rows"),
            "checksum": record.get("checksum"),
            "active": record.get("active", False),
            "ok": record.get("status") == "success",
        })
        self._save_index(idx)
        return record

    def get(self, version_id: str) -> dict[str, Any]:
        p = self.root / f"{version_id}.json"
        if not p.exists():
            raise KeyError(f"数据版本不存在: {version_id}")
        return json.loads(p.read_text(encoding="utf-8"))

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.load_index()[:limit]

    def latest_success(self) -> dict[str, Any] | None:
        for it in self.load_index():
            if it.get("status") == "success":
                return self.get(it["version_id"])
        return None

    def activate(self, version_id: str) -> dict[str, Any]:
        """把某个版本设为当前激活版本（供训练/查询默认使用）。"""
        rec = self.get(version_id)
        idx = self.load_index()
        for it in idx:
            it["active"] = (it["version_id"] == version_id)
        self._save_index(idx)
        for it in idx:
            try:
                r = self.get(it["version_id"])
                if r.get("active") != it["active"]:
                    r["active"] = it["active"]
                    (self.root / f"{r['version_id']}.json").write_text(
                        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            except KeyError:
                continue
        rec["active"] = True
        return rec

    def active_version(self) -> dict[str, Any] | None:
        for it in self.load_index():
            if it.get("active"):
                try:
                    return self.get(it["version_id"])
                except KeyError:
                    continue
        return self.latest_success()

    def delete(self, version_id: str) -> None:
        p = self.root / f"{version_id}.json"
        if p.exists():
            p.unlink()
        self._save_index([i for i in self.load_index() if i.get("version_id") != version_id])


def stage_record(
    name: str,
    rows_in: int,
    rows_out: int,
    params: dict[str, Any],
    seconds: float,
    notes: str = "",
) -> dict[str, Any]:
    """构造一条阶段记录，统一各阶段日志格式。"""
    return {
        "stage": name,
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "dropped": int(rows_in - rows_out),
        "params": params,
        "seconds": round(float(seconds), 3),
        "notes": notes,
        "at": now_str(),
    }


versions = VersionRegistry()
