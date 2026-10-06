"""
Streamlit 端的公共工具：找对的 Python 解释器、确保 FastAPI 中间服务在跑、调 Agent、画结果。

为什么需要它：
- Streamlit 可能被不同 conda 环境启动，后台子进程（FastAPI / MCP Server）必须用同一个解释器；
- 演示时最容易翻车的就是"忘了起 FastAPI"，这里在页面加载时自动拉起并探活，失败也只在侧边栏黄色提示，不崩页面。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (  # noqa: E402
    API_BASE_URL, API_HOST, API_PORT, DATASET_TEMP_DIR, DEST_BINS, YOLO_CONF_THRES,
)

# ---------------- 解释器 & 服务 ----------------
_LOG_PATH = DATASET_TEMP_DIR / "api_server.log"


def python_bin() -> str:
    """当前 Streamlit 所在解释器；优先 conda 环境里的 python。"""
    exe = Path(sys.executable)
    candidate = exe.parent / "python"
    return str(candidate if candidate.exists() else exe)


def api_alive(timeout: float = 1.5) -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("ok") is True
    except Exception:
        return False


def kill_port_occupier(port: int = API_PORT) -> str:
    """端口被别的进程占着（不是我们的服务）时，尝试清掉，返回说明文本。"""
    try:
        out = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, timeout=5)
        pids = [p for p in out.stdout.split() if p.strip()]
        if not pids:
            return f"端口 {port} 没有占用进程"
        for pid in pids:
            subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
        time.sleep(0.8)
        return f"已结束占用端口 {port} 的进程: {', '.join(pids)}"
    except Exception as e:
        return f"清理端口失败: {e}"


def ensure_api_server(port: int = API_PORT, wait: float = 25.0,
                      log_path: Path = _LOG_PATH) -> tuple[bool, str]:
    """
    确保 FastAPI 中间服务在跑。
    返回 (是否可用, 说明文本)。已经在跑就直接返回，不会重复拉起。
    """
    if api_alive():
        return True, f"FastAPI 已在运行：{API_BASE_URL}"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_bin(), "-m", "uvicorn", "api_server.arm_fastapi_server:app",
           "--host", API_HOST, "--port", str(port)]
    try:
        with open(log_path, "ab") as log:
            subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception as e:
        return False, f"启动 FastAPI 失败：{e}"

    t0 = time.time()
    while time.time() - t0 < wait:
        if api_alive():
            return True, f"已自动启动 FastAPI：{API_BASE_URL}（日志 {log_path.name}）"
        time.sleep(0.6)
    return False, (f"FastAPI 启动超时。请手动执行：\n"
                   f"  cd {ROOT} && {python_bin()} -m uvicorn api_server.arm_fastapi_server:app --port {port}\n"
                   f"  日志：{log_path}")


def sim_url(embed: bool = True, auto_pick: bool = True) -> str:
    """浏览器机械臂页面地址（带自动执行参数）。"""
    url = f"{API_BASE_URL}/sim/"
    params = []
    if embed:
        params.append("embed=1")
    if auto_pick:
        params.append("auto=1")
    return url + ("?" + "&".join(params) if params else "")


# ---------------- Agent 调用 ----------------
def run_image_inspection(image_path: str | Path, use_llm: bool = True,
                         execute: bool = True, model: str = "",
                         timeout: float = 300.0,
                         conf_thres: float = YOLO_CONF_THRES) -> dict:
    """
    跑一次"单张图片"的完整 Agent 闭环（LangGraph）。
    execute=False 时只做感知 + 决策，不动机械臂（界面上可以"先看报告再执行"）。
    """
    from agent.inspect_agent import inspect_image
    return inspect_image(str(image_path), use_llm=use_llm, execute=execute,
                         model=model, timeout=timeout, conf_thres=conf_thres)


def run_full_patrol() -> dict:
    """跑一轮完整的多点巡检（agent.agent_main 的 LangGraph 网格巡检）。"""
    from agent.agent_main import run_once
    return {"result": run_once()}


# ---------------- 小工具 ----------------
def save_upload(file_bytes: bytes, suffix: str = ".jpg") -> Path:
    """把上传的图片落到 dataset_temp/，返回路径。"""
    name = f"upload_{int(time.time() * 1000)}{suffix or '.jpg'}"
    p = DATASET_TEMP_DIR / name
    p.write_bytes(file_bytes)
    return p


def bus_state() -> Optional[dict]:
    try:
        return requests.get(f"{API_BASE_URL}/api/state", timeout=3).json()
    except Exception:
        return None


def arm_http(action: str, payload: Optional[dict] = None) -> dict[str, Any]:
    """界面上的手动操作：直接打 FastAPI 接口（不走 MCP，保证即使 MCP 异常也能演示）。"""
    payload = payload or {}
    try:
        if action == "home":
            requests.post(f"{API_BASE_URL}/api/go_home", timeout=5)
            return {"ok": True, "msg": "已下发回零"}
        if action == "sort":
            r = requests.post(f"{API_BASE_URL}/api/pick", json={
                "x": payload.get("x", 0.0), "y": payload.get("y", 0.0),
                "dest_x": (payload.get("dest_xy") or [None, None])[0],
                "dest_y": (payload.get("dest_xy") or [None, None])[1],
                "note": payload.get("note", "manual"),
            }, timeout=5)
            return {"ok": r.status_code == 200, "msg": "已下发分拣任务"}
        if action == "running":
            requests.post(f"{API_BASE_URL}/api/set_running",
                          params={"running": int(bool(payload.get("running", True)))}, timeout=5)
            return {"ok": True, "msg": "巡检开关已更新"}
    except Exception as e:
        return {"ok": False, "msg": f"操作失败：{e}"}
    return {"ok": False, "msg": f"未知操作 {action}"}


__all__ = [
    "ROOT", "python_bin", "api_alive", "ensure_api_server", "kill_port_occupier",
    "sim_url", "run_image_inspection", "run_full_patrol", "save_upload",
    "bus_state", "arm_http", "DEST_BINS", "YOLO_CONF_THRES", "API_BASE_URL", "DATASET_TEMP_DIR",
]
