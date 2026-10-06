"""
项目统一入口 / 启动说明。

真正的服务分三个终端跑（见 README.md），这个 main.py 负责：
  python main.py            打印启动指引
  python main.py --ui       直接拉起 Streamlit 交互主界面（会自动带起 FastAPI）
  python main.py --api      只启动 FastAPI 中间服务 + 浏览器机械臂页面
  python main.py --demo     直接跑一轮 LangGraph 网格巡检（需先起 FastAPI）
  python main.py --image <图片>   对单张图片跑一次完整 Agent 闭环（感知→决策→MCP 分拣）

推荐用法：`python main.py --ui`，然后浏览器打开 http://localhost:8501
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config.settings import API_BASE_URL, STREAMLIT_PORT, API_PORT

ROOT = Path(__file__).resolve().parent


def print_help():
    print(f"""
============================================================
 钢材缺陷检测与分拣 Agent - 启动指引
============================================================

【推荐】一条命令起交互主界面（内部自动拉起 FastAPI）：
    python main.py --ui
    浏览器打开:  http://localhost:{STREAMLIT_PORT}/

【手动分三个终端】
 [终端 1] FastAPI 中间通信服务 + 浏览器虚拟机械臂
    uvicorn api_server.arm_fastapi_server:app --port {API_PORT} --reload
    浏览器打开:  {API_BASE_URL}/sim/

 [终端 2] Streamlit 交互主界面（上传图片 / 摄像头检测 → 决策 → 跳转机械臂）
    streamlit run ui_streamlit/streamlit_app.py
    浏览器打开:  http://localhost:{STREAMLIT_PORT}/

 [终端 3] 跑巡检或单图检测
    python main.py --demo                      # 一轮完整网格巡检
    python main.py --image samples/demo_defect_high.jpg   # 单图闭环

【前置条件】
 - Ollama 已启动且 pulled 模型：  OLLAMA_ORIGINS='*' ollama serve
 - conda 环境：conda activate langgraph （依赖见 requirements.txt）
============================================================
""")


def run_ui():
    """拉起 Streamlit 主界面（FastAPI 会由界面自己 ensure_api_server 自动带起）。"""
    cmd = [sys.executable, "-m", "streamlit", "run",
           str(ROOT / "ui_streamlit" / "streamlit_app.py"),
           "--server.port", str(STREAMLIT_PORT)]
    print(f"[main] 启动 Streamlit: {' '.join(cmd)}")
    print(f"[main] 打开 http://localhost:{STREAMLIT_PORT}/  （Ctrl+C 退出）")
    subprocess.run(cmd, cwd=str(ROOT))


def run_api():
    cmd = [sys.executable, "-m", "uvicorn", "api_server.arm_fastapi_server:app",
           "--port", str(API_PORT), "--reload"]
    print(f"[main] 启动 FastAPI: {' '.join(cmd)}")
    print(f"[main] 机械臂页面 {API_BASE_URL}/sim/   接口文档 {API_BASE_URL}/docs")
    subprocess.run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--ui" in args:
        run_ui()
    elif "--api" in args:
        run_api()
    elif "--image" in args:
        i = args.index("--image")
        if i + 1 >= len(args):
            print("用法: python main.py --image <图片路径>")
            raise SystemExit(1)
        from agent.inspect_agent import inspect_image
        res = inspect_image(args[i + 1], use_llm="--no-llm" not in args,
                            execute="--no-exec" not in args)
        import json
        print(json.dumps({k: v for k, v in res.items() if k != "tool_calls"},
                         ensure_ascii=False, indent=2))
        print("MCP 工具调用：")
        for c in res.get("tool_calls", []):
            print(f"  - {c['tool']}({c['args']}) -> ok={c['ok']} {c['ms']}ms")
    elif "--demo" in args:
        from agent.agent_main import run_once
        run_once()
    else:
        print_help()
