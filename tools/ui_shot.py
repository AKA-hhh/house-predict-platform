#!/usr/bin/env python
"""
UI 截图工具（开发辅助）
========================
通过 Chrome DevTools Protocol 驱动 Edge/Chrome 无头浏览器，
对系统的各个页面做**真实交互后**的截图，用于 README 与文档配图。

之所以不用 `msedge --screenshot` 直接截：
  `--screenshot` 只能拿到页面初始状态，像「价格预测」「房源推荐」这些
  必须先填表单、点按钮才会出结果的页面只能截到空态。
  本工具用 CDP 下发 JS 完成填表与点击，等结果渲染后再截图。

用法：
    python tools/ui_shot.py                      # 截取全部页面
    python tools/ui_shot.py --only overview,predict
    python tools/ui_shot.py --base http://127.0.0.1:8000 --out docs/images

依赖：websockets（已随 uvicorn[standard] 安装）
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

BROWSER_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser() -> str:
    for p in BROWSER_CANDIDATES:
        if Path(p).exists():
            return p
    found = shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")
    if found:
        return found
    raise SystemExit("未找到 Edge / Chrome，请先安装或修改 BROWSER_CANDIDATES")


# ---------------------------------------------------------------- CDP 客户端
class CDP:
    """极简 CDP 客户端：只实现本条流程需要的方法。"""

    def __init__(self, ws) -> None:
        self.ws = ws
        self._id = 0

    async def call(self, method: str, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=90))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method} 失败: {msg['error']}")
                return msg.get("result", {})
            # 其余为事件消息，忽略

    async def js(self, expression: str, wait_promise: bool = True):
        r = await self.call("Runtime.evaluate", expression=expression,
                            awaitPromise=wait_promise, returnByValue=True)
        return r.get("result", {}).get("value")

    async def goto(self, url: str, settle: float = 3.0) -> None:
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(settle)

    async def shot(self, path: Path, full_page: bool = False) -> None:
        r = await self.call("Page.captureScreenshot",
                            format="png", captureBeyondViewport=full_page)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(r["data"]))


def wait_debug_port(port: int, timeout: float = 40.0) -> str:
    """等待调试端口就绪，返回页面目标的 WebSocket 地址。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                targets = json.loads(r.read())
            for t in targets:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.6)
    raise SystemExit("等待浏览器调试端口超时")


# ---------------------------------------------------------------- 交互脚本
# 每个页面：URL hash、截图前等待秒数、可选的前置 JS（填表 / 点击）
PAGES: dict[str, dict] = {
    "overview": {"hash": "#overview", "wait": 5},
    "pipeline": {"hash": "#pipeline", "wait": 3},
    "versions": {"hash": "#versions", "wait": 3},
    "quality": {"hash": "#quality", "wait": 3.5},
    "listings": {"hash": "#listings", "wait": 3.5},
    "algorithms": {"hash": "#algorithms", "wait": 3},
    "training": {"hash": "#training", "wait": 3.5},
    "predict": {
        "hash": "#predict", "wait": 3.5,
        "pre": """
            document.querySelector('#f-area').value = '89';
            document.querySelector('#f-year').value = '2005';
            document.querySelector('#f-comm').value = '68000';
            document.querySelector('#f-district').value = '朝阳区';
            document.querySelector('#pr-run').click();
        """,
        "after": 4,
    },
    "recommend": {
        "hash": "#recommend", "wait": 4.5,
        "pre": """
            document.querySelector('#r-algo').value = 'hybrid';
            document.querySelector('#r-algo').dispatchEvent(new Event('change'));
            document.querySelector('#r-district').value = '海淀区';
            document.querySelector('#r-bmax').value = '900';
            document.querySelector('#r-amin').value = '70';
            document.querySelector('#r-renov').value = '精装';
            document.querySelector('#r-run').click();
        """,
        "after": 12,
    },
    "sim": {
        "hash": "#recommend", "wait": 4,
        "pre": """
            document.querySelector('#sim-id').value = '101084782030';
            document.querySelector('#sim-run').click();
            document.querySelector('#sim-result').scrollIntoView({block:'center'});
        """,
        "after": 4,
    },
    "experiments": {"hash": "#experiments", "wait": 3},
    "docs": {"hash": "#docs", "wait": 3.5},
}


async def run(base: str, out_dir: Path, only: list[str] | None, width: int, height: int,
              full_page: bool) -> None:
    import websockets  # 延迟导入，便于给出友好报错

    browser = find_browser()
    port = 9333
    profile = Path(tempfile.mkdtemp(prefix="ui_shot_"))
    proc = subprocess.Popen([
        browser, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", "--force-device-scale-factor=1",
        f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
        f"--window-size={width},{height}", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        ws_url = wait_debug_port(port)
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            cdp = CDP(ws)
            await cdp.call("Page.enable")
            await cdp.call("Runtime.enable")
            await cdp.call("Emulation.setDeviceMetricsOverride",
                           width=width, height=height, deviceScaleFactor=1, mobile=False)

            names = only or list(PAGES.keys())
            for name in names:
                cfg = PAGES.get(name)
                if not cfg:
                    print(f"  跳过未知页面: {name}")
                    continue
                url = f"{base}/{cfg['hash']}"
                print(f"  · {name:<12} {url}")
                await cdp.goto(url, settle=cfg.get("wait", 3))
                if cfg.get("pre"):
                    await cdp.js(cfg["pre"])
                    await asyncio.sleep(cfg.get("after", 3))
                # 等图表与网络请求稳定
                await cdp.js("new Promise(r => setTimeout(r, 800))")
                await cdp.shot(out_dir / f"{name}.png", full_page=full_page)
                print(f"      → {(out_dir / f'{name}.png').relative_to(BASE_DIR)}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="系统 UI 截图工具")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--only", default="", help="逗号分隔的页面名，默认全部")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1180)
    ap.add_argument("--full-page", action="store_true", help="截取整页（含滚动区域）")
    args = ap.parse_args()

    out_dir = (BASE_DIR / args.out).resolve()
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    print(f"输出目录: {out_dir}")
    asyncio.run(run(args.base, out_dir, only, args.width, args.height, args.full_page))
    print("完成")


if __name__ == "__main__":
    main()
