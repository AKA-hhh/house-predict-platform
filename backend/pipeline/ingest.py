"""
Stage 1 · 数据导入（Ingest）
=============================
职责：
  1. 解析数据源（本地文件 / URL 直链 / SQLite 表）
  2. **自动探测文件编码**（本项目原始数据为 GB18030，直接按 UTF-8 读会乱码）
  3. **分块（chunk）流式读取**，边读边做行数统计与类型强制，控制内存峰值
  4. 列名规范化（原始中文/缩写列名 -> 规范列名）
  5. 产出导入报告（编码、行数、列数、耗时、分块信息）

设计说明（大数据视角）：
  将「读取」实现为分块扫描而非一次性 load，便于后续替换为分布式文件系统分区读取；
  行数统计与类型转换在扫描过程中增量完成，内存占用与文件大小解耦。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..settings import DEFAULT_CHUNK_SIZE, ENCODING_CANDIDATES, RAW_DIR
from .schema import COLUMN_MAP, NUMERIC_COLUMNS
from .sources import DataSource, registry


# ------------------------------------------------------------------ 编码
def detect_encoding(path: Path, probe_bytes: int = 1 << 16) -> str:
    """通过逐字节试探候选编码，返回第一个可完整解码的编码名。"""
    raw = path.open("rb").read(probe_bytes)
    for enc in ENCODING_CANDIDATES:
        try:
            raw.decode(enc)
            # 避免 latin-1 这种"永远能解码"的兜底编码抢答
            if enc == "latin-1" and any(b > 127 for b in raw[:2000]):
                continue
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _count_rows(path: Path, encoding: str) -> int:
    """流式统计行数（含表头），用于导入前后的一致性校验。"""
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return max(n - 1, 0)


# --------------------------------------------------------------- 读取器
def iter_chunks(
    path: Path,
    encoding: str,
    delimiter: str = ",",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """按块迭代读取 CSV，全部以字符串读入，由下游显式转换类型。"""
    reader = pd.read_csv(
        path,
        encoding=encoding,
        sep=delimiter,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "NaN", "null", "NULL", "None", "nan"],
        on_bad_lines="skip",
        low_memory=True,
    )
    for chunk in reader:
        yield chunk


def download_url_source(src: DataSource, force: bool = False) -> Path:
    """URL 数据源落盘（已存在且非强制则跳过）。"""
    import urllib.request

    dst = Path(registry.resolved_path(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force and dst.stat().st_size > 0:
        return dst
    req = urllib.request.Request(src.uri, headers={"User-Agent": "house-predict/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dst.open("wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
    return dst


# ----------------------------------------------------------- 列名规范化
def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """原始列名 -> 规范列名；未在映射表中的列原样保留（prefix=raw_）。"""
    mapping: dict[str, str] = {}
    rename: dict[str, str] = {}
    for col in df.columns:
        clean = str(col).strip().lstrip("\ufeff")
        if clean in COLUMN_MAP:
            rename[col] = COLUMN_MAP[clean]
            mapping[clean] = COLUMN_MAP[clean]
        else:
            rename[col] = f"raw_{clean}"
    return df.rename(columns=rename), mapping


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """把 schema 声明的数值列强制转换为 float64，非法值 -> NaN（交由清洗阶段处理）。

    统一使用 float64 而非保留 int64 的原因：后续清洗会向这些列写入 NaN（如异常值截断、
    地理坐标越界置空），而 int64 列无法容纳缺失值。
    """
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


# ----------------------------------------------------------- 主入口
def ingest(src: DataSource, chunk_size: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """执行数据导入，返回 (规范化后的 DataFrame, 导入报告)。"""
    t0 = time.time()
    report: dict[str, Any] = {"source_id": src.source_id, "source_type": src.source_type}

    if src.source_type == "url_csv":
        path = download_url_source(src)
        report["downloaded_to"] = str(path)
    elif src.source_type == "local_csv":
        path = Path(registry.resolved_path(src))
    elif src.source_type == "sqlite":
        from ..settings import DB_PATH
        df = pd.read_sql(f"SELECT * FROM {src.uri}", f"sqlite:///{DB_PATH}")
        report.update({"rows": len(df), "encoding": "n/a", "chunks": 1})
        df, mapping = normalize_columns(df)
        report["column_map"] = mapping
        report["seconds"] = round(time.time() - t0, 3)
        return df, report
    else:
        raise ValueError(f"不支持的数据源类型: {src.source_type}")

    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    encoding = src.encoding
    if encoding in ("auto", "", None):
        encoding = detect_encoding(path)

    csize = int(chunk_size or src.options.get("chunk_size") or DEFAULT_CHUNK_SIZE)

    frames: list[pd.DataFrame] = []
    scanned = 0
    n_chunks = 0
    for chunk in iter_chunks(path, encoding, src.delimiter, csize):
        chunk, mapping = normalize_columns(chunk)
        chunk = coerce_numeric(chunk)
        frames.append(chunk)
        scanned += len(chunk)
        n_chunks += 1
        if not report.get("column_map"):
            report["column_map"] = mapping

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    report.update({
        "file_path": str(path),
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "encoding": encoding,
        "chunk_size": csize,
        "chunks": n_chunks,
        "rows": int(len(df)),
        "raw_line_rows": _count_rows(path, encoding),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "seconds": round(time.time() - t0, 3),
        "source_attribution": src.attribution,
        "source_license": src.license,
    })
    return df, report


def profile(df: pd.DataFrame) -> dict[str, Any]:
    """列级数据画像：dtype / 缺失数 / 缺失率 / 唯一值数 / 样例。"""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_rate": round(float(s.isna().mean()), 4),
            "n_unique": int(s.nunique(dropna=True)),
            "sample": " | ".join(str(v) for v in s.dropna().head(3).tolist())[:80],
        })
    rows.sort(key=lambda r: r["missing_rate"], reverse=True)
    return {"rows": int(len(df)), "columns": int(df.shape[1]), "profile": rows}


__all__ = [
    "ingest", "profile", "detect_encoding", "iter_chunks",
    "normalize_columns", "coerce_numeric", "download_url_source", "RAW_DIR",
]
