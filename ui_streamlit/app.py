"""
Streamlit 巡检看板（监控视角）。

和 streamlit_app.py 的区别：
- streamlit_app.py  = 交互主界面（上传图片/摄像头检测 → 决策 → 跳机械臂页面）；
- 本文件           = 全局看板（完整网格巡检、SQLite 日志、总线状态、MCP 工具）。

两者互不冲突，浏览器里可以各开一个标签页。
启动：streamlit run ui_streamlit/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_streamlit import common as C  # noqa: E402
from config.settings import API_BASE_URL, DATASET_TEMP_DIR  # noqa: E402

st.set_page_config(page_title="钢材缺陷巡检看板", layout="wide")
st.title("📊 钢材表面缺陷巡检看板（LangGraph + 本地大模型 + 虚拟机械臂）")

# ---------- 服务自检 ----------
if "api_ready" not in st.session_state:
    st.session_state.api_ready, st.session_state.api_msg = C.ensure_api_server()
API_OK = C.api_alive()

with st.sidebar:
    st.header("控制台")
    if API_OK:
        st.success(f"FastAPI 在线\n\n{API_BASE_URL}", icon="✅")
    else:
        st.error("FastAPI 不可用", icon="⚠️")
        st.caption(st.session_state.get("api_msg", ""))
        if st.button("🔄 重新拉起服务", width='stretch'):
            st.session_state.api_ready, st.session_state.api_msg = C.ensure_api_server()
            st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ 启动巡检", width='stretch'):
            with st.spinner("正在跑完整网格巡检（需要机械臂页面已打开）…"):
                try:
                    C.run_full_patrol()
                    st.success("巡检完成 ✅")
                except Exception as e:  # noqa: BLE001
                    st.error(f"巡检失败：{e}")
    with c2:
        if st.button("⏹️ 停止", width='stretch'):
            C.arm_http("running", {"running": False})
            st.toast("已关闭巡检总线", icon="⏹️")

    st.divider()
    st.markdown(f"[🦾 打开机械臂页面]({C.sim_url(embed=False)})")
    st.markdown(f"[FastAPI 调试文档]({API_BASE_URL}/docs)")
    st.caption("交互式单图检测请运行：`streamlit run ui_streamlit/streamlit_app.py`")

# ---------- 主区 ----------
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("📷 最新视觉检测")
    bus = C.bus_state()
    frame = (bus or {}).get("latest_frame")
    if frame:
        vis = DATASET_TEMP_DIR / f"vis_{frame}"
        if vis.exists():
            st.image(str(vis), caption=f"YOLO 检测可视化: {frame}", width='stretch')
        else:
            st.image(f"{API_BASE_URL}/api/frame/{frame}", caption=frame, width='stretch')
    else:
        st.info("还没有收到虚拟相机帧。请先打开机械臂页面（它负责上传虚拟相机截图）。")

    if bus and bus.get("decision"):
        d = bus["decision"]
        st.markdown(
            f"**最新决策**（来源：{bus.get('decision_source') or '-'}）："
            f"`{d.get('action')}` → {d.get('dest_label') or '—'}　{d.get('reason') or ''}"
        )

with col2:
    st.subheader("📜 巡检日志（最近 15 条）")
    try:
        from mcp_service.mcp_logger_server import logger
        rows = logger.query_recent(limit=15)
    except Exception as e:  # noqa: BLE001
        rows = []
        st.warning(f"日志读取失败：{e}")
    if rows:
        st.dataframe(
            rows, width='stretch',
            column_config={
                "ts": "时间", "grid_index": "网格点", "has_defect": "有缺陷",
                "defect_cls": "类别", "conf": "置信度", "llm_action": "动作",
                "detail": "理由",
            },
        )
    else:
        st.write("暂无日志。")

st.divider()


@st.fragment(run_every=2.0)
def _bus_monitor():
    st.subheader(f"🔌 总线状态（Agent ↔ 浏览器机械臂）　`{datetime.now().strftime('%H:%M:%S')}`")
    st_ = C.bus_state()
    if st_:
        st.json(st_)
    else:
        st.write("未连接。")


_bus_monitor()
