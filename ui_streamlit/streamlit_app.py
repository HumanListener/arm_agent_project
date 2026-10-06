"""
Streamlit 交互层 —— 多模态工业缺陷检测与分拣 Agent 的操作台。

页面结构：
  1. 缺陷检测（主页面）：
     - 上传图片 / 打开摄像头拍照（浏览器端拍照，不需要本机摄像头也可用上传方式）；
     - 跑 YOLO 感知 → 本地 Ollama 决策 → 通过 MCP 工具驱动浏览器里的机械臂分拣；
     - 检测到缺陷时给出「打开机械臂页面」跳转按钮，并可直接把页面嵌在下方同屏演示。
  2. 巡检看板：完整网格巡检（agent_main 的 LangGraph）+ SQLite 日志 + 总线状态。
  3. MCP 工具面板：现场 list_tools / call_tool，证明执行层走的是 MCP 协议。

启动：streamlit run ui_streamlit/streamlit_app.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_streamlit import common as C  # noqa: E402
from config.settings import (  # noqa: E402
    API_BASE_URL, DATASET_TEMP_DIR, DEST_BINS, YOLO_CONF_THRES, YOLO_CONF_MIN, YOLO_CONF_MAX,
)

st.set_page_config(page_title="工业缺陷检测与分拣 Agent", page_icon="🏭", layout="wide")


def _defect_cn(cls: str) -> str:
    """英文缺陷类别 → 中文名（导入失败时原样返回）。"""
    if not cls:
        return ""
    try:
        from agent.decision import defect_cn
        return defect_cn(cls)
    except Exception:  # noqa: BLE001
        return cls


# ============================================================
# 0) 启动时确保中间服务在跑（演示最怕"忘了起 FastAPI"）
# ============================================================
if "api_ready" not in st.session_state:
    ok, msg = C.ensure_api_server()
    st.session_state.api_ready = ok
    st.session_state.api_msg = msg
    st.session_state.api_checked_at = datetime.now().strftime("%H:%M:%S")

API_OK = C.api_alive()
if API_OK and not st.session_state.get("api_ready"):
    st.session_state.api_ready = True
    st.session_state.api_msg = f"FastAPI 已就绪：{API_BASE_URL}"

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🏭 缺陷检测・分拣 Agent")
    st.caption("YOLO 感知 → LangGraph 决策 → MCP 驱动机械臂")

    st.subheader("🔌 系统状态")
    if API_OK:
        st.success(f"FastAPI 中间服务：在线\n\n{API_BASE_URL}", icon="✅")
    else:
        st.error(f"FastAPI 不可用\n\n{st.session_state.get('api_msg', '')}", icon="⚠️")
        if st.button("🔄 重新拉起服务", width='stretch'):
            ok, msg = C.ensure_api_server()
            st.session_state.api_ready, st.session_state.api_msg = ok, msg
            st.rerun()

    ollama_ok, models = (False, [])
    try:
        from agent.decision import ollama_available, resolve_model
        ollama_ok, models = ollama_available()
        chosen = resolve_model()
    except Exception as e:  # noqa: BLE001
        resolve_model = lambda *a, **k: ""  # noqa: E731
        chosen = ""
        st.warning(f"决策模块加载失败：{e}")
    if ollama_ok:
        st.success(f"Ollama：在线（{len(models)} 个模型）", icon="🧠")
        st.caption("当前模型：" + (chosen or "无可用模型"))
    else:
        st.warning("Ollama：离线 → 自动使用规则决策", icon="🧠")

    st.divider()
    st.subheader("⚙️ 检测参数")
    use_llm = st.toggle("调用本地大模型决策", value=True,
                        help=f"关闭则只用规则知识库，秒出结果。本地 8B 模型单次决策约 60~90s。")
    conf_thres = st.slider("YOLO 置信度阈值", float(YOLO_CONF_MIN), float(YOLO_CONF_MAX),
                           float(YOLO_CONF_THRES), 0.05,
                           help="高于此值判为缺陷；低于 DEFECT_HIGH_CONF(0.60) 的严重缺陷会转人工复检。")
    auto_open = st.toggle("检出缺陷后自动打开机械臂页面", value=True)

    st.divider()
    st.subheader("🕹️ 手动控制")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🏠 回零", width='stretch'):
            st.toast(C.arm_http("home")["msg"], icon="🏠")
    with c2:
        if st.button("↻ 刷新状态", width='stretch'):
            st.rerun()
    st.markdown(f"🔗 [机械臂页面]({C.sim_url(embed=False)})　|　[API 文档]({API_BASE_URL}/docs)")

    bus = C.bus_state() or {}
    st.caption(
        f"总线：到达={'是' if bus.get('arm_arrived') else '否'}　"
        f"最新帧={bus.get('latest_frame') or '—'}"
    )

# ============================================================
# 主区
# ============================================================
st.title("🏭 基于多模态智能体的工业缺陷检测与分拣系统")
st.caption("感知（YOLOv8 钢材质检）→ 决策（LangGraph + Ollama）→ 行动（MCP → 机械臂）")

tab_detect, tab_board, tab_mcp = st.tabs(["🔍 缺陷检测", "📊 巡检看板", "🧰 MCP 工具面板"])

# ------------------------------------------------------------
# Tab 1：缺陷检测
# ------------------------------------------------------------
with tab_detect:
    src = st.radio("图片来源", ["📁 上传图片", "🎥 摄像头拍照", "🗂️ 示例图片"],
                   horizontal=True, label_visibility="collapsed")

    image_bytes = None
    image_name = ""

    if src.startswith("📁"):
        up = st.file_uploader("上传一张钢板表面图片（jpg/png）",
                              type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            image_bytes, image_name = up.getvalue(), up.name
        else:
            st.info("💡 手边没有缺陷图？切到「🗂️ 示例图片」可以直接用内置样例跑通整条链路。")
    elif src.startswith("🎥"):
        shot = st.camera_input("用摄像头拍一张（浏览器会请求摄像头权限）")
        if shot is not None:
            image_bytes, image_name = shot.getvalue(), f"camera_{int(time.time())}.jpg"
        st.caption("提示：没有摄像头也没关系，切回「上传图片」或「示例图片」即可；"
                   "权限被拒时同样用上传方式。")
    else:
        samples = sorted((ROOT / "samples").glob("*.jpg")) if (ROOT / "samples").exists() else []
        if not samples:
            st.warning("samples/ 目录为空，请先放入几张钢板图片。")
        else:
            picks = st.multiselect(
                "选择示例图片（可多选，逐个跑一遍）",
                [p.name for p in samples],
                default=[samples[0].name],
                help="demo_defect_high.jpg = 高置信缺陷（会真的驱动机械臂分拣）；"
                     "demo_defect.jpg = 低置信度（转人工复检）；demo_clean.jpg = 无缺陷放行",
            )
            if picks:
                image_bytes = (ROOT / "samples" / picks[0]).read_bytes()
                image_name = picks[0]
                st.session_state["pending_sample_batch"] = picks
            st.caption("示例图片路径：`samples/`（可以用 `python tools/make_samples.py` 重新生成）")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        run_detect = st.button("🚀 开始检测", type="primary", width='stretch',
                               disabled=image_bytes is None)
    with col_btn2:
        run_dry = st.button("🧪 只出报告（不动机械臂）", width='stretch',
                            disabled=image_bytes is None)
    with col_btn3:
        st.caption("检测 → 决策 → 分拣 全链路自动执行；「只出报告」用于先看决策再决定动手。")

    if image_bytes is not None:
        st.image(image_bytes, caption=f"输入图像：{image_name}", width=420)

    if run_detect or run_dry:
        path = C.save_upload(image_bytes, Path(image_name).suffix or ".jpg")
        execute = bool(run_detect)
        with st.spinner("① YOLO 感知中…"):
            time.sleep(0.1)
        try:
            with st.spinner("② 决策中…（调用本地大模型可能需要 60~90 秒，可关闭侧边栏开关加速）"
                            if use_llm else "② 规则决策中…"):
                t0 = time.time()
                result = C.run_image_inspection(path, use_llm=use_llm, execute=execute,
                                                conf_thres=conf_thres)
                result["elapsed"] = round(time.time() - t0, 1)
            st.session_state["last_result"] = result
            st.session_state["last_executed"] = execute
            st.session_state["last_input"] = str(path)
        except Exception as e:  # noqa: BLE001
            st.error(f"检测流程异常：{e}")
            st.exception(e)

    res = st.session_state.get("last_result")
    if res:
        det = res.get("detect", {})
        dec = res.get("decision", {})
        executed = st.session_state.get("last_executed", False)
        has_defect = bool(det.get("has_defect"))

        st.divider()
        if has_defect:
            st.error(f"🔴 **检测到缺陷**：{det.get('defect_cls')}　置信度 {det.get('conf')}", icon="🔴")
        else:
            st.success("✅ 未检测到缺陷，工件可直接放行", icon="✅")

        left, right = st.columns([1.05, 1], gap="large")
        cn_full = _defect_cn(det.get("defect_cls") or "")
        defect_cn_suffix = f"（{cn_full}）" if cn_full and cn_full != det.get("defect_cls") else ""

        with left:
            st.subheader("🔍 视觉检测结果")
            vis = res.get("vis_path")
            if vis and Path(vis).exists():
                st.image(vis, caption="YOLO 检测可视化", width='stretch')
            elif st.session_state.get("last_input"):
                st.image(st.session_state["last_input"], width='stretch')

            st.markdown(
                f"- **是否缺陷**：{'是' if has_defect else '否'}\n"
                f"- **缺陷类别**：{det.get('defect_cls') or '—'}{defect_cn_suffix}\n"
                f"- **置信度**：{det.get('conf')}\n"
                f"- **像素框 xyxy**：{det.get('bbox_xyxy')}\n"
                f"- **工作台世界坐标 (x, y)**：{det.get('world_xy')}\n"
                f"- **推理耗时**：{res.get('elapsed', '—')} s"
            )
            all_dets = det.get("all_detections") or []
            if len(all_dets) > 1:
                with st.expander(f"全部候选框（{len(all_dets)} 个）"):
                    st.dataframe(
                        [{"类别": d["cls"], "置信度": round(d["conf"], 3), "框": d["bbox_xyxy"]}
                         for d in all_dets],
                        width='stretch',
                    )

        with right:
            st.subheader("🧠 Agent 决策")
            engine = dec.get("engine", "rule")
            tag = "本地大模型" if engine.startswith("ollama") else "规则知识库"
            st.markdown(
                f"- **执行动作**：`{dec.get('action', '—')}`\n"
                f"- **投放区域**：{dec.get('dest_label') or '无需投放（放行）'}\n"
                f"- **严重度**：{dec.get('severity', '—')}\n"
                f"- **决策来源**：{tag}（{engine}）\n"
                f"- **判据**：{dec.get('reason', '—')}"
            )
            st.text_area("决策报告", value=dec.get("report", ""), height=170, disabled=True)

            tool_calls = res.get("tool_calls") or []
            with st.expander(f"🦾 MCP 工具调用轨迹（{len(tool_calls)} 次）", expanded=bool(tool_calls)):
                if not tool_calls:
                    st.caption("本次只做决策，未调用机械臂工具。")
                for c in tool_calls:
                    icon = "✅" if c.get("ok") else "❌"
                    st.markdown(f"{icon} `{c['tool']}({c['args']})` — {c['ms']} ms")
                    with st.popover("查看返回"):
                        st.json(c.get("result"))

            # ---- 缺陷 → 跳转浏览器机械臂页面 ----
            if has_defect:
                st.divider()
                st.markdown("#### 🦾 去浏览器机械臂页面分拣")
                sim_url = C.sim_url(embed=True, auto_pick=True)
                if executed:
                    st.info("已通过 MCP 向机械臂下发分拣任务。"
                            "打开机械臂页面后会自动执行抓取 → 投放 → 回零。", icon="📤")
                    if st.session_state.pop("auto_opened_pending", False):
                        st.session_state["open_sim_now"] = True
                else:
                    st.warning("当前为「只出报告」模式，尚未驱动机械臂。", icon="🧪")
                    if st.button("🦾 现在执行分拣", type="primary", width='stretch'):
                        wx, wy = det.get("world_xy") or [0.0, 0.0]
                        bin_xy = DEST_BINS.get(dec.get("dest") or "ng", DEST_BINS["ng"])["xy"]
                        out = C.arm_http("sort", {"x": wx, "y": wy, "dest_xy": bin_xy,
                                                  "note": f"manual {det.get('defect_cls')}"})
                        st.toast(out["msg"], icon="🦾")
                        st.session_state["auto_opened_pending"] = True
                        st.rerun()

                st.markdown(
                    f'<a href="{sim_url}" target="_blank" rel="noopener">'
                    f'<button style="width:100%;padding:10px;border-radius:8px;border:none;'
                    f'background:#1f6feb;color:#fff;font-size:15px;cursor:pointer;">'
                    f'🚀 打开浏览器机械臂页面（自动执行分拣）</button></a>',
                    unsafe_allow_html=True,
                )
                st.caption("若浏览器拦截了自动跳转，请点击上面的按钮；页面里的动作也会在下方同屏显示。")

                show_embed = st.toggle("📺 在页面内同屏显示机械臂（便于演示/答辩）", value=False,
                                       key="embed_sim")
                if show_embed:
                    st.components.v1.iframe(sim_url, height=620, scrolling=False)
                    st.caption("上方就是浏览器里的虚拟机械臂（Three.js），它正在轮询总线执行分拣任务。")

    # 自动打开机械臂页面（浏览器可能拦截，故同时提供按钮）
    sim_url_open = C.sim_url(embed=False, auto_pick=True)
    if st.session_state.pop("open_sim_now", False):
        st.components.v1.html(
            f"<script>window.open('{sim_url_open}', '_blank');</script>", height=0
        )
        st.toast("已尝试打开机械臂页面，若没反应请点上面的蓝色按钮", icon="🚀")

# ------------------------------------------------------------
# Tab 2：巡检看板
# ------------------------------------------------------------
with tab_board:
    st.subheader("📊 自动巡检（虚拟相机 + 网格点位闭环）")
    colA, colB = st.columns([1, 2])
    with colA:
        st.caption("需要浏览器机械臂页面处于打开状态（它负责按请求上传虚拟相机截图）。")
        if st.button("▶️ 启动一轮完整巡检", type="primary", width='stretch'):
            with st.spinner("正在跑 LangGraph 网格巡检…"):
                try:
                    C.run_full_patrol()
                    st.success("巡检完成 ✅")
                except Exception as e:  # noqa: BLE001
                    st.error(f"巡检失败：{e}")
        if st.button("⏹️ 停止巡检总线", width='stretch'):
            C.arm_http("running", {"running": False})
            st.toast("已关闭巡检总线", icon="⏹️")
        if st.button("▶️ 打开巡检总线", width='stretch'):
            C.arm_http("running", {"running": True})
            st.toast("已打开巡检总线", icon="▶️")
    with colB:
        st.markdown("**最近一次虚拟相机帧**")
        try:
            frame = (C.bus_state() or {}).get("latest_frame")
        except Exception:  # noqa: BLE001
            frame = None
        if frame:
            p = DATASET_TEMP_DIR / frame
            if p.exists():
                st.image(str(p), caption=frame, width=380)
            else:
                st.image(f"{API_BASE_URL}/api/frame/{frame}", width=380)
        else:
            st.info("还没有帧。请先打开机械臂页面 → 点「启动一轮完整巡检」。")

    st.divider()
    st.subheader("📜 巡检日志（SQLite / MCP 工具写入）")
    try:
        from mcp_service.mcp_logger_server import logger
        rows = logger.query_recent(limit=30)
    except Exception as e:  # noqa: BLE001
        rows = []
        st.warning(f"日志读取失败：{e}")
    if rows:
        st.dataframe(
            rows, width='stretch',
            column_config={
                "id": "ID", "ts": "时间", "grid_index": "网格点", "has_defect": "有缺陷",
                "defect_cls": "类别", "conf": "置信度", "world_xy": "世界坐标",
                "llm_action": "动作", "detail": "理由",
            },
        )
    else:
        st.caption("暂无日志。")

# ------------------------------------------------------------
# Tab 3：MCP 工具面板
# ------------------------------------------------------------
with tab_mcp:
    st.subheader("🧰 执行层：MCP 机械臂工具服务")
    st.markdown(
        "Agent 的「动手能力」来自 MCP Server：`mcp_service/mcp_arm_tools.py`。"
        "下面这个面板用标准 MCP 客户端（stdio）连上去，列出工具并现场调用，"
        "可以看到整条链路：**MCP 工具 → HTTP 总线 → 浏览器虚拟机械臂**。"
    )
    colL, colR = st.columns([1, 1], gap="large")
    with colL:
        if st.button("🔍 列出 MCP 工具", width='stretch'):
            with st.spinner("连接 MCP server…"):
                try:
                    from mcp_service.mcp_client import list_arm_tools
                    st.session_state["mcp_tools"] = list_arm_tools()
                except Exception as e:  # noqa: BLE001
                    st.error(f"连接 MCP server 失败：{e}")
        tools = st.session_state.get("mcp_tools") or []
        for t in tools:
            st.markdown(f"- **`{t['name']}`** — {(t['description'] or '').splitlines()[0]}")
        if tools:
            st.caption(f"共 {len(tools)} 个工具（MCP 1.x，stdio 传输）")
    with colR:
        st.markdown("**现场调用工具**")
        tool_name = st.selectbox(
            "工具", ["get_arm_state", "go_home", "move_arm_to", "request_camera_capture",
                     "sort_defect", "publish_decision", "set_running"],
            index=0,
        )
        defaults = {
            "get_arm_state": "{}",
            "go_home": "{}",
            "move_arm_to": '{"x": 0.10, "y": 0.05, "z": 0.05, "note": "面板测试"}',
            "request_camera_capture": '{"timeout": 15}',
            "sort_defect": '{"x": 0.10, "y": 0.05, "dest": "ng", "note": "面板测试"}',
            "publish_decision": '{"action": "SORT", "reason": "面板测试", "dest_label": "废品区"}',
            "set_running": '{"running": true}',
        }
        args_text = st.text_area("参数（JSON）", value=defaults.get(tool_name, "{}"), height=100)
        if st.button("📞 调用 MCP 工具", type="primary", width='stretch'):
            import json
            try:
                args = json.loads(args_text or "{}")
            except Exception as e:  # noqa: BLE001
                st.error(f"参数不是合法 JSON：{e}")
                args = None
            if args is not None:
                with st.spinner(f"调用 {tool_name} …（可能等待机械臂页面响应）"):
                    from mcp_service.mcp_client import call_arm_tool
                    t0 = time.time()
                    out = call_arm_tool(tool_name, args, timeout=60)
                st.success(f"调用完成，用时 {time.time() - t0:.1f}s")
                st.json(out)
        st.caption("提示：`sort_defect` / `go_home` 只有在浏览器机械臂页面打开时才会真正到位。")

# 底部：自动刷新总线状态（用 fragment 局部刷新，不打断用户操作）
st.divider()


@st.fragment(run_every=2.0)
def _bus_monitor():
    st.markdown(f"**🔌 总线实时状态**  `{datetime.now().strftime('%H:%M:%S')}`")
    st_ = C.bus_state()
    if st_:
        cols = st.columns(4)
        cols[0].metric("机械臂到位", "是" if st_.get("arm_arrived") else "否")
        cols[1].metric("最新帧", (st_.get("latest_frame") or "—")[:18])
        cols[2].metric("待执行分拣", "有" if st_.get("pick_task") else "无")
        cols[3].metric("巡检总线", "开" if st_.get("running") else "关")
        if st_.get("decision"):
            st.caption(f"最新决策（{st_.get('decision_source') or '-'}）："
                       f"{st_['decision'].get('action')} "
                       f"{st_['decision'].get('dest_label') or ''} — "
                       f"{st_['decision'].get('reason') or ''}")
    else:
        st.caption("连不上 FastAPI 中间服务。")


_bus_monitor()
