"""
机械臂 MCP Server —— 把"控制机械臂"这件事封装成标准 MCP 工具，供 Agent 调用。

这是项目"执行层 (Action)"的核心：
  Agent(LangGraph) --MCP(stdio)--> mcp_arm_tools --> HTTP --> FastAPI bus --> 浏览器虚拟机械臂

暴露的工具：
  get_arm_state            查询机械臂/bus 当前状态
  move_arm_to              末端移动到工作台某个坐标（世界坐标，米）
  go_home                  回零
  request_camera_capture   让虚拟相机拍一张并上传，返回帧文件名
  sort_defect              完整分拣动作：抓取缺陷工件 → 投放目标区 → 回零
  set_running              开/停巡检总线

单独调试：
  python -m mcp_service.mcp_arm_tools            # stdio 模式（给 MCP 客户端）
  python -m mcp_service.mcp_arm_tools --http     # streamable-http 模式，便于 curl 调试
  python -m mcp_service.mcp_arm_tools --selftest # 直接跑一遍工具函数（不起 MCP 协议）
"""
from __future__ import annotations

import asyncio
import sys
from mcp.server.fastmcp import FastMCP

from api_server.arm_client import arm
from config.settings import DEST_BINS

mcp = FastMCP("arm-tools")


# ---------------- 工具实现 ----------------
async def _to_thread(fn, *args, **kw):
    """MCP 工具是 async 的，但底层是 requests 同步调用，丢到线程池避免阻塞事件循环。"""
    return await asyncio.to_thread(fn, *args, **kw)


@mcp.tool()
async def get_arm_state() -> dict:
    """查询虚拟机械臂与通信总线的当前状态。

    返回字段：target_pose(当前目标点[米])、arm_arrived(是否已到位)、
    latest_frame(最新相机帧文件名)、frame_ts(帧时间戳)、pick_task(待执行分拣任务)。
    """
    st = await _to_thread(arm.state)
    if st is None:
        return {"ok": False, "error": "连不上 FastAPI 中间服务(http://127.0.0.1:8000)，"
                                      "请先启动：uvicorn api_server.arm_fastapi_server:app --port 8000"}
    st["ok"] = True
    return st


@mcp.tool()
async def move_arm_to(x: float, y: float, z: float = 0.05, note: str = "") -> dict:
    """把机械臂末端移动到工作台坐标 (x, y)，高度 z（单位：米）。

    工作台范围大约 x∈[-0.3, 0.3]，y∈[-0.2, 0.2]，z=0.05 为贴近工件高度。
    """
    x = max(-0.30, min(0.30, float(x)))
    y = max(-0.20, min(0.20, float(y)))
    z = max(0.01, min(0.30, float(z)))
    r = await _to_thread(arm.move_to, x, y, z, note)
    if r is None:
        return {"ok": False, "error": "下发点位失败：机械臂中间服务不可用"}
    arrived = await _to_thread(arm.wait_arrived, 15.0)
    return {"ok": True, "target": [x, y, z], "arrived": arrived,
            "hint": "" if arrived else "浏览器机械臂页面未打开或未上报到位"}


@mcp.tool()
async def go_home() -> dict:
    """机械臂回到安全归零位（HOME）。"""
    r = await _to_thread(arm.go_home)
    if r is None:
        return {"ok": False, "error": "归零失败：机械臂中间服务不可用"}
    await _to_thread(arm.wait_arrived, 12.0)
    return {"ok": True, "target": [0.0, 0.0, 0.15]}


@mcp.tool()
async def request_camera_capture(timeout: float = 15.0) -> dict:
    """请浏览器端的虚拟相机拍一张照片并上传，返回新帧的文件名。

    需要浏览器已经打开虚拟机械臂页面（/sim/），否则会超时。
    """
    await _to_thread(arm.request_capture)
    r = await _to_thread(arm.wait_new_frame, 0.0, float(timeout))
    if not r:
        return {"ok": False, "error": f"{timeout}s 内没等到新帧，请确认浏览器已打开虚拟机械臂页面 /sim/"}
    return {"ok": True, "frame": r.get("frame"), "frame_ts": r.get("frame_ts")}


@mcp.tool()
async def sort_defect(x: float, y: float, dest: str = "ng", note: str = "") -> dict:
    """对坐标 (x, y) 处的缺陷工件执行完整分拣：

    移动到缺陷位置 → 夹爪闭合抓取 → 抬起 → 移动到目标区 → 松开放下 → 回零。

    dest 指定投放区域，可选：
      "ng"     → 废品区  (-0.25, 0.15)  严重缺陷直接报废
      "rework" → 返修区  ( 0.25, 0.15)  轻微缺陷打磨返修
      "review" → 复检区  ( 0.00, -0.18) 不确定，交人工复检
    """
    bin_info = DEST_BINS.get(dest) or DEST_BINS["ng"]
    dx, dy = bin_info["xy"]
    task = await _to_thread(arm.sort_defect, float(x), float(y), [dx, dy],
                            note or f"sort->{bin_info['label']}")
    if task is None:
        return {"ok": False, "error": "分拣指令下发失败：机械臂中间服务不可用"}
    arrived = await _to_thread(arm.wait_arrived, 20.0)
    return {"ok": True, "pick": [x, y], "dest": bin_info["label"], "dest_xy": [dx, dy],
            "arrived": arrived,
            "hint": "" if arrived else "已下发，但浏览器未在上报到位（页面可能没打开）"}


@mcp.tool()
async def set_running(running: bool) -> dict:
    """打开/关闭自动巡检总线开关。"""
    r = await _to_thread(arm.set_running, running)
    if r is None:
        return {"ok": False, "error": "设置失败：机械臂中间服务不可用"}
    return r


@mcp.tool()
async def publish_decision(action: str, reason: str, dest_label: str = "") -> dict:
    """把 Agent 的决策结论广播到总线，浏览器页面 HUD 会同步显示（可留痕、便于答辩演示）。"""
    r = await _to_thread(arm.sync_decision,
                         {"action": action, "reason": reason, "dest_label": dest_label},
                         {}, "agent")
    return r or {"ok": False, "error": "广播失败：机械臂中间服务不可用"}


# ---------------- 入口 ----------------
def run_server(http: bool = False, port: int = 8765):
    if http:
        print(f"[MCP] arm-tools server (streamable-http) 启动: http://127.0.0.1:{port}/mcp")
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
    else:
        print("[MCP] arm-tools server 启动（stdio 模式）", file=sys.stderr)
        mcp.run(transport="stdio")


def _selftest():
    """不经过 MCP 协议，直接调工具函数，用于本地快速验证执行层链路。"""
    async def _main():
        print("state  :", (await get_arm_state()).get("ok"))
        print("home   :", await go_home())
        print("move   :", await move_arm_to(0.1, 0.05, 0.05))
        print("sort   :", await sort_defect(0.1, 0.05, "ng"))
        print("publish:", await publish_decision("PICK", "自检：链路连通", "废品区"))
    asyncio.run(_main())


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        run_server(http="--http" in sys.argv)
