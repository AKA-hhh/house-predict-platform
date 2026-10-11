#!/usr/bin/env python
"""
二手房价格分析与预测系统 —— 启动脚本
=====================================
用法：
    python run.py                 # 默认 http://127.0.0.1:8000
    python run.py --port 9000
    python run.py --init          # 先跑一次完整数据流水线再启动服务
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))


def main() -> None:
    parser = argparse.ArgumentParser(description="二手房价格分析与预测系统")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--init", action="store_true", help="启动前先运行一次数据流水线")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    if args.init:
        from backend.pipeline.runner import run_pipeline
        print(">>> 正在运行数据流水线 …")
        rec = run_pipeline("lianjia_bj_raw", {})
        print(f">>> 数据版本 {rec['version_id']} 完成，最终 {rec['final_rows']} 行")

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"\n  二手房价格分析与预测系统\n  服务地址: {url}\n  接口文档: {url}/docs\n")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
