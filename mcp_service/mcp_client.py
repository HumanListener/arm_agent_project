"""
极简 MCP 客户端：用 stdio 拉起 mcp_service/mcp_arm_tools.py，列出/调用里面的工具。

用途：
- agent/inspect_agent.py：真正走"Agent 调 MCP 工具"的路径（而不是直接 import 函数）；
- ui_streamlit/streamlit_app.py：界面上的「MCP 工具面板」，可以现场 list_tools 并调用，
  直观展示"大模型通过 MCP 协议驱动机械臂"这件事。

用法：
    from mcp_service.mcp_client import list_arm_tools, call_arm_tool
    print(list_arm_tools())                      # [{'name': 'go_home', ...}, ...]
    print(call_arm_tool("go_home", {}))
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

from config.settings import ARM_MCP_SERVER, PROJECT_ROOT, ARM_TOOL_TIMEOUT

MCP_PYTHON = sys.executable


# ---------------- 底层：建立一次会话 ----------------
def _server_params(server_path=ARM_MCP_SERVER):
    from mcp import StdioServerParameters

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return StdioServerParameters(
        command=MCP_PYTHON,
        args=[str(server_path)],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


async def _with_session(fn, server_path=ARM_MCP_SERVER):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _server_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=ARM_TOOL_TIMEOUT)
            return await fn(session)


def _run(coro):
    """在同步代码里跑一次 MCP 会话；已经处于事件循环中时用独立线程跑。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


# ---------------- 同步 API ----------------
def list_arm_tools() -> list[dict]:
    """列出 MCP 机械臂服务暴露的所有工具。"""
    async def _inner(session):
        res = await session.list_tools()
        return [{"name": t.name, "description": (t.description or "").strip(),
                 "schema": getattr(t, "inputSchema", None)} for t in res.tools]

    try:
        return _run(_with_session(_inner))
    except Exception as e:
        print(f"[MCP] list_tools 失败: {e}")
        return []


def call_arm_tool(name: str, arguments: Optional[dict] = None,
                  timeout: float = ARM_TOOL_TIMEOUT) -> dict:
    """
    调用一个 MCP 工具，返回解析后的结果 dict（工具本身返回 JSON 文本）。
    失败时返回 {"ok": False, "error": "..."}，不抛异常。
    """
    arguments = arguments or {}

    async def _inner(session):
        res = await asyncio.wait_for(
            session.call_tool(name, arguments=arguments), timeout=timeout
        )
        texts = []
        for c in getattr(res, "content", []) or []:
            txt = getattr(c, "text", None)
            if txt:
                texts.append(txt)
        raw = "\n".join(texts)
        if getattr(res, "isError", False):
            return {"ok": False, "error": raw or "MCP 工具执行出错", "raw": raw}
        try:
            return json.loads(raw)
        except Exception:
            return {"ok": True, "raw": raw}

    try:
        return _run(_with_session(_inner))
    except Exception as e:
        return {"ok": False, "error": f"MCP 调用 {name} 失败: {e}"}


def call_arm_tool_safe(name: str, arguments: Optional[dict] = None, **kw) -> dict:
    """带兜底：MCP 不可用时返回 error 结构，方便 Streamlit 直接展示。"""
    return call_arm_tool(name, arguments, **kw)


if __name__ == "__main__":
    tools = list_arm_tools()
    print(f"共 {len(tools)} 个 MCP 工具:")
    for t in tools:
        print(f"  - {t['name']}: {t['description'].splitlines()[0] if t['description'] else ''}")
    if len(sys.argv) > 1:
        tool = sys.argv[1]
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        print(f"\n调用 {tool}({args}) →")
        print(json.dumps(call_arm_tool(tool, args), ensure_ascii=False, indent=2))
