"""
LangGraph 具身巡检 Agent（多点网格自动巡检）。

状态图节点：
  capture   - 通知浏览器虚拟相机拍照，等待上传新帧（并把机械臂挪到该网格点上方）
  perceive  - YOLO 推理 + OpenCV 预处理
  decide    - 统一决策引擎（规则知识库 + 本地 Ollama），输出 PICK / REVIEW / RESHOOT / NEXT
  act       - PICK：浏览器执行完整分拣动作；REVIEW：悬停提示人工复检
  next_grid - 记录日志并前进到下一个网格点
  done      - 回零结束

条件路由：
  RESHOOT -> capture（置信度极低，重拍）
  PICK/REVIEW -> act
  NEXT    -> next_grid / done

说明：
- 单张图片的交互式闭环在 agent/inspect_agent.py（Streamlit 用那个）；
- 本文件是"自动巡检"模式，需要浏览器打开 /sim/ 页面来提供虚拟相机帧；
- Ollama 未启动时自动降级为规则决策，保证闭环可演示。
"""
from __future__ import annotations

import time
from typing import TypedDict, Optional

import cv2
import requests

from config.settings import (
    API_BASE_URL, INSPECT_GRID, YOLO_CONF_THRES, USE_LLM,
    TOOL_Z_PICK, TOOL_Z_HOME, DATASET_TEMP_DIR,
)
from yolo_module.yolo_defect_infer import detector
from mcp_service.mcp_logger_server import logger


# ============ 1. 与 FastAPI 总线通信的客户端 ============
class ArmSimClient:
    """Agent 侧的总线客户端：薄封装，失败不抛异常（演示时机械臂页面没开也不至于中断巡检）。"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base = base_url.rstrip("/")

    def _get(self, path: str, timeout: float = 3.0) -> dict:
        try:
            r = requests.get(f"{self.base}{path}", timeout=timeout)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def _post(self, path: str, timeout: float = 3.0, **kw) -> dict:
        try:
            r = requests.post(f"{self.base}{path}", timeout=timeout, **kw)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def current_frame_ts(self) -> float:
        return float(self._get("/api/latest_frame").get("frame_ts") or 0.0)

    def request_capture(self):
        """请浏览器拍一张并上传（清旧标志 → 发请求）。"""
        self._post("/api/ack_capture")
        self._post("/api/request_capture")

    def wait_new_frame(self, prev_ts: float, timeout: float = 15.0) -> Optional[str]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self._get("/api/latest_frame")
            if r.get("frame") and float(r.get("frame_ts") or 0) > prev_ts:
                return r["frame"]
            time.sleep(0.3)
        return None

    def move_to(self, x: float, y: float, z: float = TOOL_Z_PICK, timeout: float = 15.0) -> bool:
        self._post("/api/dispatch", json={"x": x, "y": y, "z": z, "note": ""})
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._get("/api/state").get("arm_arrived"):
                return True
            time.sleep(0.3)
        return False

    def sort_defect(self, x: float, y: float, dest_xy: list, timeout: float = 20.0) -> bool:
        """让浏览器执行完整"抓取 → 投放 → 回零"动作序列。"""
        self._post("/api/pick", json={"x": x, "y": y,
                                     "dest_x": dest_xy[0], "dest_y": dest_xy[1],
                                     "note": "patrol"})
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._get("/api/state").get("arm_arrived"):
                return True
            time.sleep(0.3)
        return False

    def go_home(self):
        self._post("/api/go_home")


# ============ 2. 遗留的直连函数（保留给脚本/调试用；巡检决策统一走 agent.decision）============
def ollama_chat(prompt: str) -> Optional[str]:
    """直接 HTTP 调 Ollama /api/chat（流式关掉）；失败返回 None。"""
    from agent.decision import ollama_chat as _chat
    return _chat(prompt)


def rule_decision(detect: dict) -> dict:
    """规则兜底决策（现在统一由 agent.decision.rule_decide 提供）。"""
    from agent.decision import rule_decide
    return rule_decide(detect)


# ============ 3. LangGraph 状态 ============
class AgentState(TypedDict, total=False):
    grid_index: int            # 当前巡检网格下标
    prev_frame_ts: float        # 上次帧时间戳，用于等新帧
    frame_name: str             # 最新上传帧文件名
    detect: dict                # YOLO 检测结果
    llm_decision: dict          # 大模型/规则决策
    round_log: str


sim = ArmSimClient()


# ============ 4. 节点 ============
def node_capture(state: AgentState) -> AgentState:
    """通知浏览器拍一张照片，等待新帧。"""
    idx = state.get("grid_index", 0)
    # 先移动到下一个网格点正上方（俯视拍照位）
    gx, gy = INSPECT_GRID[idx]
    prev_ts = sim.current_frame_ts()          # 先记录当前帧时间戳，避免把旧帧当新帧
    sim.move_to(gx, gy, TOOL_Z_HOME)
    sim.request_capture()
    frame = sim.wait_new_frame(prev_ts)
    if frame is None:
        return {"round_log": "TIMEOUT 等不到浏览器帧（请确认机械臂页面 /sim/ 已打开）"}
    return {"frame_name": frame, "prev_frame_ts": sim.current_frame_ts()}


def node_perceive(state: AgentState) -> AgentState:
    frame = state.get("frame_name")
    if not frame:
        return {"detect": {"has_defect": False, "defect_cls": "", "conf": 0.0,
                           "bbox_xyxy": [], "world_xy": None}}
    img = cv2.imread(str(DATASET_TEMP_DIR / frame))
    if img is None:
        return {"detect": {"has_defect": False, "defect_cls": "", "conf": 0.0,
                           "bbox_xyxy": [], "world_xy": None}}
    vis = DATASET_TEMP_DIR / f"vis_{frame}"
    result = detector.detect(img, save_vis_path=vis)
    return {"detect": result}


# 巡检点位 → 投放区：用 decision 模块里同一套规则，保证和 Streamlit 侧结论一致
def _dest_for(detect: dict) -> tuple[str, str]:
    from agent.decision import rule_decide
    from config.settings import DEST_BINS
    dec = rule_decide(detect)
    dest = dec.get("dest") or "review"
    return dest, DEST_BINS.get(dest, DEST_BINS["review"])["label"]


def node_decide(state: AgentState) -> AgentState:
    """
    决策节点：复用 agent/decision.py 的统一决策引擎（规则知识库 + 本地 Ollama）。
    PICK   → 去抓取分拣
    REVIEW → 置信度不足，悬停提示人工复检
    RESHOOT/NEXT → 重拍 / 下一个网格点
    """
    from agent.decision import decide as decide_unified
    det = state["detect"]
    dec = decide_unified(det, use_llm=USE_LLM)
    action_map = {"SORT": "PICK", "REVIEW": "REVIEW", "PASS": "NEXT"}
    action = action_map.get(dec.get("action"), "NEXT")
    # 置信度极低时先重拍一次，而不是直接放行
    if not det.get("has_defect") and 0 < float(det.get("conf") or 0) < YOLO_CONF_THRES:
        action = "RESHOOT"
    out = {
        "action": action,
        "reason": dec.get("reason", ""),
        "dest": dec.get("dest", ""),
        "dest_label": dec.get("dest_label", ""),
        "engine": dec.get("engine", "rule"),
        "report": dec.get("report", ""),
    }
    print(f"[Decide] {out['action']} <- {out['engine']} :: {out['reason']}")
    return {"llm_decision": out}


def route_after_decide(state: AgentState) -> str:
    act = state["llm_decision"].get("action", "NEXT")
    idx = state.get("grid_index", 0)
    if act == "RESHOOT":
        return "capture"
    if act in ("PICK", "REVIEW"):
        return "act"
    # NEXT：走下一个网格
    if idx + 1 >= len(INSPECT_GRID):
        return "done"
    return "next_grid"


def node_act(state: AgentState) -> AgentState:
    """
    动作节点：
    - PICK   → 让浏览器执行完整分拣动作（抓取 → 投放目标区 → 回零）；
    - REVIEW → 只悬停到缺陷上方并回零，提示人工复检，不抓取。
    """
    det = state["detect"]
    dec = state["llm_decision"]
    idx = state.get("grid_index", 0)
    wx, wy = det.get("world_xy") or INSPECT_GRID[idx]
    dest, dest_label = _dest_for(det)

    if dec.get("action") == "REVIEW":
        sim.move_to(wx, wy, TOOL_Z_HOME)
        sim.go_home()
        ok = True
        dest_label = "复检区（悬停提示）"
    else:
        from config.settings import DEST_BINS
        ok = sim.sort_defect(wx, wy, DEST_BINS.get(dest, DEST_BINS["review"])["xy"])

    detail = (f"{dec.get('reason', '')} | 投放={dest_label} | "
              f"{'到位' if ok else '未收到浏览器到位回报'}")
    logger.write_log(
        grid_index=idx,
        has_defect=det.get("has_defect", False),
        defect_cls=det.get("defect_cls", ""),
        conf=det.get("conf", 0.0),
        world_xy=[wx, wy],
        llm_action=dec.get("action", ""),
        detail=detail,
    )
    return {"round_log": f"{dec.get('action')} ({wx},{wy}) → {dest_label}"}


def node_next_grid(state: AgentState) -> AgentState:
    idx = state.get("grid_index", 0)
    det = state["detect"]
    logger.write_log(
        grid_index=idx,
        has_defect=False,
        defect_cls=det.get("defect_cls", ""),
        conf=det.get("conf", 0.0),
        world_xy=det.get("world_xy"),
        llm_action="NEXT",
        detail=state["llm_decision"].get("reason", ""),
    )
    return {"grid_index": idx + 1}


def node_done(state: AgentState) -> AgentState:
    sim.go_home()
    return {"round_log": "巡检完成"}


# ============ 5. 组装状态图 ============
def build_graph():
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(AgentState)
    g.add_node("capture", node_capture)
    g.add_node("perceive", node_perceive)
    g.add_node("decide", node_decide)
    g.add_node("act", node_act)
    g.add_node("next_grid", node_next_grid)
    g.add_node("done", node_done)

    g.add_edge(START, "capture")
    g.add_edge("capture", "perceive")
    g.add_edge("perceive", "decide")
    g.add_conditional_edges(
        "decide", route_after_decide,
        {"capture": "capture", "act": "act", "next_grid": "next_grid", "done": "done"},
    )
    g.add_edge("act", "next_grid")
    g.add_edge("next_grid", "capture")
    g.add_edge("done", END)
    return g.compile()


agent = build_graph()


def run_once():
    """跑一整轮完整巡检（从 grid 0 到末尾）。"""
    init = {"grid_index": 0, "prev_frame_ts": 0.0}
    for step in agent.stream(init, stream_mode="values"):
        gi = step.get("grid_index", 0)
        log = step.get("round_log", "")
        det = step.get("detect", {})
        dec = step.get("llm_decision", {})
        print(f"[Agent] grid={gi} detect={det.get('defect_cls')} conf={det.get('conf')} "
              f"action={dec.get('action')} log={log}")
    return "巡检结束"


if __name__ == "__main__":
    run_once()
