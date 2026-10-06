"""
全链路自检脚本：一次跑完"感知 → 决策 → MCP 执行 → FastAPI 总线"四层，
用来确认环境没坏、答辩前先跑一遍。

用法：
    python tools/verify_all.py                # 全链路（含 MCP 真实调用，需要 FastAPI 在跑）
    python tools/verify_all.py --no-arm       # 跳过会真的驱动机械臂的项
    python tools/verify_all.py --with-llm     # 额外验证本地大模型决策（8B CPU 约 60~90s）

输出每一项 [PASS]/[FAIL]，最后给出汇总；退出码非 0 表示有失败项。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn, critical: bool = True):
    t0 = time.time()
    try:
        detail = fn() or ""
        ok = True
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    RESULTS.append((name, ok, str(detail)))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}  ({time.time() - t0:.2f}s)  {detail}")
    return ok


def main():
    no_arm = "--no-arm" in sys.argv
    with_llm = "--with-llm" in sys.argv

    print("=" * 70)
    print(" 钢材缺陷检测与分拣 Agent —— 全链路自检")
    print("=" * 70)

    # ---------- 1. 配置 ----------
    def _config():
        from config.settings import (API_BASE_URL, DEST_BINS, YOLO_CONF_THRES,
                                     ARM_MCP_SERVER, DB_PATH)
        assert ARM_MCP_SERVER.exists(), f"MCP Server 不存在: {ARM_MCP_SERVER}"
        assert len(DEST_BINS) == 3, "DEST_BINS 应为 3 个投放区"
        return (f"API={API_BASE_URL} 阈值={YOLO_CONF_THRES} 投放区="
                f"{[v['label'] for v in DEST_BINS.values()]} DB={DB_PATH.name}")

    check("配置加载", _config)

    # ---------- 2. 服务 ----------
    def _api():
        import requests
        from config.settings import API_BASE_URL
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        assert r.status_code == 200 and r.json().get("ok"), r.text
        return f"{API_BASE_URL} 在线"

    api_ok = check("FastAPI 指令总线", _api, critical=False)

    def _sim_page():
        import requests
        from config.settings import API_BASE_URL
        r = requests.get(f"{API_BASE_URL}/sim/?embed=1&auto=1", timeout=5)
        assert r.status_code == 200
        html = r.text
        assert "ikArm" in html and "pickAndPlace" in html and "arm_ready" in html
        return "机械臂页面可访问，IK / 抓取 / 到位回调代码齐全"

    if api_ok:
        check("浏览器机械臂页面", _sim_page, critical=False)

    # ---------- 3. 感知层 ----------
    def _yolo():
        import cv2
        from yolo_module.yolo_defect_infer import detector
        from config.settings import DATASET_TEMP_DIR
        img = cv2.imread(str(ROOT / "samples" / "demo_defect_high.jpg"))
        assert img is not None, "示例图片读取失败"
        vis = DATASET_TEMP_DIR / "verify_vis.jpg"
        r = detector.detect(img, save_vis_path=vis)
        assert r["has_defect"], f"高置信样例应判为缺陷，实际 {r}"
        assert vis.exists(), "可视化图未生成"
        return (f"{r['defect_cls']} conf={r['conf']} world_xy={r['world_xy']} "
                f"类别数={len(detector._model.names) if detector._model else 'mock'}")

    check("感知层 YOLO 推理 + 可视化", _yolo)

    def _clean_pass():
        import cv2
        from yolo_module.yolo_defect_infer import detector
        img = cv2.imread(str(ROOT / "samples" / "demo_clean.jpg"))
        r = detector.detect(img)
        assert not r["has_defect"], f"无缺陷样例不应判为缺陷：{r}"
        return f"无缺陷样例正确放行（最高置信度 {r['conf']}）"

    check("感知层 无缺陷样例放行", _clean_pass)

    # ---------- 4. 决策层 ----------
    def _rule():
        from agent.decision import rule_decide
        cases = [
            ({"has_defect": True, "defect_cls": "inclusion", "conf": 0.91}, "SORT", "ng"),
            ({"has_defect": True, "defect_cls": "crazing", "conf": 0.70}, "SORT", "rework"),
            ({"has_defect": True, "defect_cls": "scratches", "conf": 0.30}, "REVIEW", "review"),
            ({"has_defect": False, "defect_cls": "", "conf": 0.0}, "PASS", ""),
            ({"has_defect": True, "defect_cls": "unknown_class", "conf": 0.80}, "REVIEW", "review"),
        ]
        for det, act, dest in cases:
            d = rule_decide(det)
            assert d["action"] == act and d["dest"] == dest, f"{det} → {d}"
            assert d["report"] and d["dest_label"] is not None
        return f"{len(cases)} 条规则用例（含未知类别兜底）结论全部正确"

    check("决策层 规则知识库", _rule)

    def _dest_consistency():
        """config 的投放区坐标必须和浏览器页面里的 DEST_BINS 完全一致。"""
        import re
        from config.settings import DEST_BINS
        html = (ROOT / "sim_web_static" / "index.html").read_text()
        block = re.search(r"const DEST_BINS = \{(.*?)\};", html, re.S).group(1)
        for key, info in DEST_BINS.items():
            m = re.search(rf"{key}:\s*\{{[^}}]*x:\s*(-?[\d.]+),\s*z:\s*(-?[\d.]+)", block)
            assert m, f"页面里找不到 {key} 的坐标"
            px, pz = float(m.group(1)), float(m.group(2))
            cx, cy = info["xy"]
            assert abs(px - cx) < 1e-6 and abs(pz - cy) < 1e-6, \
                f"{key} 坐标不一致：页面({px},{pz}) vs 配置({cx},{cy})"
        return "Python 配置与浏览器页面 DEST_BINS 坐标严格一致"

    check("决策层 投放区坐标一致性", _dest_consistency)

    if with_llm:
        def _llm():
            from agent.decision import decide, resolve_model
            model = resolve_model()
            assert model, "Ollama 无可用模型"
            det = {"has_defect": True, "defect_cls": "inclusion", "conf": 0.91,
                   "bbox_xyxy": [10, 10, 60, 60], "world_xy": [0.01, 0.0]}
            d = decide(det, use_llm=True, model=model)
            assert d["engine"].startswith("ollama"), f"未走大模型：{d['engine']}"
            return f"模型={model} 用时={d.get('llm_ms')}ms 结论={d['action']}/{d['dest_label']}"

        check("决策层 Ollama 大模型", _llm, critical=False)

    # ---------- 5. 执行层：MCP ----------
    def _mcp_list():
        from mcp_service.mcp_client import list_arm_tools
        tools = list_arm_tools()
        names = {t["name"] for t in tools}
        need = {"get_arm_state", "move_arm_to", "go_home", "sort_defect", "request_camera_capture"}
        missing = need - names
        assert not missing, f"缺少工具: {missing}"
        return f"{len(tools)} 个 MCP 工具：{sorted(names)}"

    mcp_ok = check("执行层 MCP 工具枚举（stdio 协议）", _mcp_list)

    if mcp_ok and api_ok:
        def _mcp_state():
            from mcp_service.mcp_client import call_arm_tool
            r = call_arm_tool("get_arm_state", {})
            assert r.get("ok"), r
            return f"总线状态读取成功（arm_arrived={r.get('arm_arrived')}）"

        check("执行层 MCP 调用 get_arm_state", _mcp_state, critical=False)

        if not no_arm:
            def _mcp_sort():
                from mcp_service.mcp_client import call_arm_tool
                r = call_arm_tool("sort_defect", {"x": 0.01, "y": 0.0, "dest": "ng",
                                                  "note": "verify_all"}, timeout=60)
                assert r.get("ok"), r
                note = "已到位" if r.get("arrived") else "已下发（浏览器页面未打开，任务已挂在总线上）"
                return f"投放={r.get('dest')} {note}"

            check("执行层 MCP 调用 sort_defect", _mcp_sort, critical=False)

    # ---------- 6. LangGraph ----------
    def _graph_inspect():
        from agent.inspect_agent import inspect_image
        res = inspect_image(str(ROOT / "samples" / "demo_defect_high.jpg"),
                            use_llm=False, execute=False)
        assert res["decision"]["action"] == "SORT", res["decision"]
        assert res["vis_path"] and Path(res["vis_path"]).exists()
        return f"单图闭环：{res['summary']}"

    check("LangGraph 单图闭环（决策，不动机械臂）", _graph_inspect)

    def _graph_patrol():
        from agent.agent_main import agent
        nodes = set(agent.get_graph().nodes)
        need = {"capture", "perceive", "decide", "act", "next_grid", "done"}
        assert need <= nodes, f"状态图缺节点: {need - nodes}"
        return f"网格巡检状态图节点齐全：{sorted(need)}"

    check("LangGraph 网格巡检状态图", _graph_patrol)

    # ---------- 7. 日志 ----------
    def _logger():
        from mcp_service.mcp_logger_server import logger
        rid = logger.write_log(grid_index=-99, has_defect=True, defect_cls="selfcheck",
                              conf=0.99, world_xy=[0.0, 0.0], llm_action="VERIFY",
                              detail="tools/verify_all.py 自检写入")
        rows = logger.query_recent(limit=3)
        assert any(r["id"] == rid for r in rows), "刚写入的日志查不到"
        return f"SQLite 读写正常（id={rid}）"

    check("MCP 日志服务（SQLite 持久化）", _logger)

    # ---------- 汇总 ----------
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 70)
    print(f" 共 {len(RESULTS)} 项，通过 {len(RESULTS) - len(failed)} 项"
          + (f"，失败 {len(failed)} 项：{failed}" if failed else "，全部通过 ✅"))
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
