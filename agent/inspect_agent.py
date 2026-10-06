"""
单张图片的具身 Agent 闭环（LangGraph）—— Streamlit「上传图片 / 摄像头拍照」检测走的就是这条链。

    perceive(YOLO)  →  decide(规则 + 本地 Ollama)  →  act(通过 MCP 调用机械臂工具)

与 agent_main.py 的区别：
- agent_main.py 是"多点网格自动巡检"（虚拟相机 → 浏览器上传帧 → 9 个点循环）；
- 本文件是"用户给一张图"，是交互式、单次闭环，供 Streamlit 界面调用。

关键点：act 节点不是直接 import 机械臂函数，而是走 mcp_service.mcp_client 真正发起 MCP 工具调用，
        并在 state["tool_calls"] 里留下完整调用轨迹（界面上会原样展示，证明"Agent 通过 MCP 动手"）。
"""
from __future__ import annotations

import time
from typing import TypedDict

import cv2

from config.settings import DATASET_TEMP_DIR, TOOL_Z_HOME, YOLO_CONF_THRES
from yolo_module.yolo_defect_infer import detector
from agent.decision import decide as make_decision, defect_cn
from mcp_service.mcp_client import call_arm_tool
from mcp_service.mcp_logger_server import logger


# ============ 状态 ============
class InspectState(TypedDict, total=False):
    image_path: str              # 输入图片
    use_llm: bool                # 是否调用本地大模型
    execute: bool                # 是否真的让机械臂动手
    model: str                   # 指定 Ollama 模型
    conf_thres: float            # YOLO 判定阈值（界面滑块）
    detect: dict                 # YOLO 结构化结果
    decision: dict               # 决策结果
    tool_calls: list             # MCP 工具调用轨迹
    vis_path: str                # 画框可视化图路径
    tool_ok: bool                # 动作是否全部成功
    summary: str                 # 一句话总结（界面顶部展示）


# ============ 节点 1：感知 ============
def node_perceive(state: InspectState) -> InspectState:
    img = cv2.imread(state["image_path"])
    if img is None:
        return {"detect": {"has_defect": False, "defect_cls": "", "conf": 0.0,
                           "bbox_xyxy": [], "world_xy": None, "all_detections": [],
                           "error": "图片读取失败"},
                "tool_calls": list(state.get("tool_calls", []))}

    vis_path = DATASET_TEMP_DIR / f"vis_upload_{int(time.time() * 1000)}.jpg"
    det = detector.detect(img, save_vis_path=vis_path,
                          conf_thres=float(state.get("conf_thres") or YOLO_CONF_THRES))
    return {"detect": det, "vis_path": str(vis_path)}


# ============ 节点 2：决策 ============
def node_decide(state: InspectState) -> InspectState:
    det = state.get("detect", {})
    dec = make_decision(det, use_llm=state.get("use_llm", True), model=state.get("model", ""))
    return {"decision": dec}


def route_after_decide(state: InspectState) -> str:
    """根据决策选择下一跳：分拣 / 转复检 / 放行。"""
    if not state.get("execute", True):
        return "finish"
    action = (state.get("decision") or {}).get("action", "PASS")
    if action == "SORT":
        return "act_sort"
    if action == "REVIEW":
        return "act_review"
    return "finish"


# ============ 节点 3：动作（真正走 MCP）============
def _mcp(name: str, args: dict, calls: list) -> dict:
    """调用一次 MCP 工具并把轨迹记下来。"""
    t0 = time.time()
    res = call_arm_tool(name, args)
    calls.append({
        "tool": name,
        "args": args,
        "result": res,
        "ms": int((time.time() - t0) * 1000),
        "ok": bool(res.get("ok")),
    })
    print(f"[MCP] {name}({args}) -> ok={res.get('ok')} {str(res)[:160]}")
    return res


def node_act_sort(state: InspectState) -> InspectState:
    """有明确缺陷：通过 MCP 让机械臂抓走并投放到对应区域。"""
    det, dec = state["detect"], state["decision"]
    calls = list(state.get("tool_calls", []))
    wx, wy = (det.get("world_xy") or [0.0, 0.0])
    dest = dec.get("dest") or "ng"

    _mcp("publish_decision", {"action": dec.get("action", "SORT"),
                              "reason": dec.get("reason", ""),
                              "dest_label": dec.get("dest_label", "")}, calls)
    res = _mcp("sort_defect", {"x": wx, "y": wy, "dest": dest,
                               "note": f"{det.get('defect_cls')} {det.get('conf')}"}, calls)

    logger.write_log(grid_index=-1, has_defect=True,
                     defect_cls=det.get("defect_cls", ""), conf=det.get("conf", 0.0),
                     world_xy=[wx, wy], llm_action=dec.get("action", ""),
                     detail=(f"{dec.get('reason', '')} | 投放={dec.get('dest_label', '')} | "
                             f"MCP={'OK' if res.get('ok') else res.get('error')}"))
    return {
        "tool_calls": calls,
        "tool_ok": bool(res.get("ok")),
        "summary": (f"检出 {defect_cn(det.get('defect_cls', ''))}（{det.get('conf'):.2f}）→ "
                    f"已通过 MCP 分拣到「{dec.get('dest_label', '')}」"),
    }


def node_act_review(state: InspectState) -> InspectState:
    """不确定的缺陷：不判废，回零并提示人工复检（也走 MCP 留痕）。"""
    det, dec = state["detect"], state["decision"]
    calls = list(state.get("tool_calls", []))
    wx, wy = (det.get("world_xy") or [0.0, 0.0])

    _mcp("publish_decision", {"action": dec.get("action", "REVIEW"),
                              "reason": dec.get("reason", ""),
                              "dest_label": dec.get("dest_label", "复检区")}, calls)
    # 转到工件上方并回零（不抓取），示意"待人工复检"
    _mcp("move_arm_to", {"x": wx, "y": wy, "z": TOOL_Z_HOME, "note": "REVIEW 悬停"}, calls)
    _mcp("go_home", {}, calls)

    logger.write_log(grid_index=-1, has_defect=True,
                     defect_cls=det.get("defect_cls", ""), conf=det.get("conf", 0.0),
                     world_xy=[wx, wy], llm_action="REVIEW",
                     detail=dec.get("reason", ""))
    return {
        "tool_calls": calls,
        "tool_ok": True,
        "summary": (f"检出疑似 {defect_cn(det.get('defect_cls', ''))}（{det.get('conf')}）→ "
                    f"置信度不足，已提示人工复检"),
    }


def node_finish(state: InspectState) -> InspectState:
    """无缺陷 / 只做决策不动手：记录日志收尾。"""
    det = state.get("detect", {})
    dec = state.get("decision", {})
    if not dec.get("action") or dec.get("action") == "PASS":
        logger.write_log(grid_index=-1, has_defect=bool(det.get("has_defect")),
                         defect_cls=det.get("defect_cls", ""), conf=det.get("conf", 0.0),
                         world_xy=det.get("world_xy"), llm_action=dec.get("action", "PASS"),
                         detail=dec.get("reason", ""))
    if not state.get("summary"):
        return {"summary": dec.get("reason", "已完成检测")}
    return {}


# ============ 组装状态图 ============
def build_graph():
    from langgraph.graph import StateGraph, START, END

    g = StateGraph(InspectState)
    g.add_node("perceive", node_perceive)
    g.add_node("decide", node_decide)
    g.add_node("act_sort", node_act_sort)
    g.add_node("act_review", node_act_review)
    g.add_node("finish", node_finish)

    g.add_edge(START, "perceive")
    g.add_edge("perceive", "decide")
    g.add_conditional_edges("decide", route_after_decide,
                            {"act_sort": "act_sort", "act_review": "act_review", "finish": "finish"})
    g.add_edge("act_sort", "finish")
    g.add_edge("act_review", "finish")
    g.add_edge("finish", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ============ 对外入口 ============
def inspect_image(image_path: str, use_llm: bool = True, execute: bool = True,
                  model: str = "", timeout: float = 300.0,
                  conf_thres: float = YOLO_CONF_THRES) -> dict:
    """
    跑一次单图闭环，返回最终 state（含 detect / decision / tool_calls / vis_path / summary）。
    用 stream 逐步推进，方便调试时打印每一步。
    """
    init: InspectState = {
        "image_path": str(image_path), "use_llm": use_llm, "execute": execute,
        "model": model, "conf_thres": float(conf_thres), "tool_calls": [],
    }
    final: dict = dict(init)
    graph = get_graph()
    for step in graph.stream(init, stream_mode="values"):
        final.update(step)
        det, dec = step.get("detect"), step.get("decision")
        if det is not None and dec is not None:
            print(f"[Agent] 检测={det.get('defect_cls')} conf={det.get('conf')} "
                  f"决策={dec.get('action')} engine={dec.get('engine')}")
    final.setdefault("tool_calls", [])
    final.setdefault("summary", "")
    return final


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m agent.inspect_agent <图片路径> [--no-llm] [--no-exec]")
        raise SystemExit(1)
    path = sys.argv[1]
    res = inspect_image(path,
                        use_llm="--no-llm" not in sys.argv,
                        execute="--no-exec" not in sys.argv)
    import json
    print(json.dumps({k: v for k, v in res.items() if k != "tool_calls"},
                     ensure_ascii=False, indent=2))
    print("MCP 工具调用轨迹:")
    for c in res["tool_calls"]:
        print(f"  - {c['tool']}({c['args']}) -> ok={c['ok']} {c['ms']}ms {str(c['result'])[:120]}")
