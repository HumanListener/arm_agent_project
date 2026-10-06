"""
机械臂 HTTP 客户端：把 FastAPI 中间服务(bus)的 REST 接口包成一组 Python 函数。

谁会用：
- mcp_service/mcp_arm_tools.py  →  MCP Server 内部用它真正驱动浏览器里的机械臂；
- ui_streamlit/*                →  界面显示总线状态 / 手动下发点位；
- agent/agent_main.py           →  巡检闭环下发点位。

约定：所有函数都是"同步 + 带超时"，失败不抛异常（返回 ok=False 或 None），
这样 Agent 在演示环境下不会因为机械臂页面没打开直接崩。
"""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

from config.settings import API_BASE_URL, TOOL_Z_HOME, ARM_SETTLE_SEC


class ArmClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = 3.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # ---------------- 内部 ----------------
    def _get(self, path: str, **kw) -> Optional[dict]:
        try:
            r = requests.get(f"{self.base}{path}", timeout=self.timeout, **kw)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    def _post(self, path: str, **kw) -> Optional[dict]:
        try:
            r = requests.post(f"{self.base}{path}", timeout=self.timeout, **kw)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    # ---------------- 健康检查 ----------------
    def ping(self) -> bool:
        return self._get("/api/health") is not None

    def state(self) -> Optional[dict]:
        """完整总线状态（含 target_pose / arm_arrived / pending_pick / frame_ts）。"""
        return self._get("/api/state")

    # ---------------- 动作 ----------------
    def move_to(self, x: float, y: float, z: float, note: str = "") -> Optional[dict]:
        """下发一个末端目标点（世界坐标，米）。"""
        return self._post("/api/dispatch", json={"x": x, "y": y, "z": z, "note": note})

    def go_home(self) -> Optional[dict]:
        return self._post("/api/go_home")

    def sort_defect(self, x: float, y: float, dest_xy: Optional[list] = None,
                    note: str = "") -> Optional[dict]:
        """
        让浏览器端跑完整"抓取 → 投放到目标区 → 回零"动作序列。
        浏览器没有坑位信息就默认投到废品区（settings.DEST_BINS['ng']）。
        """
        return self._post("/api/pick", json={
            "x": x, "y": y,
            "dest_x": dest_xy[0] if dest_xy else None,
            "dest_y": dest_xy[1] if dest_xy else None,
            "note": note,
        })

    def sync_decision(self, decision: Optional[dict], detect: Optional[dict],
                      source: str = "streamlit") -> Optional[dict]:
        """把 Agent 的决策广播到总线，浏览器页面 HUD 可以直接显示。"""
        return self._post("/api/decision", json={
            "decision": decision or {},
            "detect": detect or {},
            "source": source,
        })

    def request_capture(self) -> Optional[dict]:
        """请浏览器拍一张虚拟相机截图并上传。"""
        return self._post("/api/request_capture")

    def wait_new_frame(self, prev_ts: float = 0.0, timeout: float = 15.0) -> Optional[dict]:
        """轮询等待一张比 prev_ts 更新的帧。返回 {frame, frame_ts}。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self._get("/api/latest_frame")
            if r and r.get("frame") and r.get("frame_ts", 0) > prev_ts:
                return r
            time.sleep(0.3)
        return None

    def wait_arrived(self, timeout: float = 15.0) -> bool:
        """等待浏览器上报 arm_ready。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.state()
            if st and st.get("arm_arrived"):
                return True
            time.sleep(0.3)
        return False

    def pick_and_place(self, x: float, y: float, dest_xy: Optional[list] = None,
                       note: str = "", timeout: float = 20.0) -> bool:
        """同步版分拣：下发动作序列后阻塞等到浏览器报告 arm_ready。"""
        if self.sort_defect(x, y, dest_xy, note) is None:
            return False
        time.sleep(ARM_SETTLE_SEC * 0.5)
        return self.wait_arrived(timeout=timeout)

    def set_running(self, running: bool) -> Optional[dict]:
        return self._post("/api/set_running", params={"running": int(bool(running))})

    def reset_home(self) -> dict[str, Any]:
        self.go_home()
        return {"ok": True, "target": [0.0, 0.0, TOOL_Z_HOME]}


# 模块级默认单例
arm = ArmClient()
