#!/usr/bin/env bash
# ============================================================================
#  一键启动：钢材缺陷检测与分拣 Agent
#
#  用法：
#      ./start.sh              # 启动 FastAPI + Streamlit，并自动打开浏览器
#      ./start.sh --no-browser # 只启动服务，不打开浏览器
#      ./start.sh --stop       # 停止本项目启动的服务
#      ./start.sh --status     # 只看当前运行状态
#
#  也可以直接在 PyCharm 里右键 Run（.idea/runConfigurations 里已配好 3 个入口）。
# ============================================================================
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

API_PORT=8000
UI_PORT=8501
LOG_DIR="$PROJECT_DIR/dataset_temp"
mkdir -p "$LOG_DIR"

# ---------- 找 Python 解释器：优先带 streamlit 的 conda 环境 ----------
PY=""
for cand in \
    "/opt/anaconda3/envs/langgraph/bin/python" \
    "/opt/anaconda3/envs/yolo/bin/python" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"
do
    [ -n "$cand" ] || continue
    [ -x "$cand" ] || continue
    if "$cand" -c "import streamlit, fastapi, cv2" >/dev/null 2>&1; then
        PY="$cand"; break
    fi
done

if [ -z "$PY" ]; then
    echo "❌ 找不到装了 streamlit/fastapi/opencv 的 Python 环境。"
    echo "   请先执行： conda activate langgraph  &&  pip install -r requirements.txt"
    exit 1
fi
echo "✅ Python 环境: $PY"

port_pid() { lsof -ti "tcp:$1" 2>/dev/null | head -1; }
is_up()    { curl -s --max-time 2 "$1" >/dev/null 2>&1; }

# ---------- --stop ----------
if [ "${1:-}" = "--stop" ]; then
    for p in $API_PORT $UI_PORT; do
        pid="$(port_pid "$p")"
        if [ -n "$pid" ]; then
            kill -9 "$pid" 2>/dev/null && echo "🛑 已停止端口 $p 上的服务 (pid $pid)"
        else
            echo "ℹ️  端口 $p 上没有运行中的服务"
        fi
    done
    exit 0
fi

# ---------- --status ----------
if [ "${1:-}" = "--status" ]; then
    is_up "http://127.0.0.1:$API_PORT/api/health" \
        && echo "✅ FastAPI  (8000) 运行中" || echo "❌ FastAPI  (8000) 未运行"
    is_up "http://127.0.0.1:$UI_PORT/_stcore/health" \
        && echo "✅ Streamlit(8501) 运行中" || echo "❌ Streamlit(8501) 未运行"
    is_up "http://localhost:11434/api/tags" \
        && echo "✅ Ollama   (11434) 运行中" || echo "⚠️  Ollama   未运行（决策会降级为规则库）"
    exit 0
fi

OPEN_BROWSER=1
[ "${1:-}" = "--no-browser" ] && OPEN_BROWSER=0

# ---------- 0) Ollama 提醒 ----------
if is_up "http://localhost:11434/api/tags"; then
    echo "✅ Ollama 在线"
else
    echo "⚠️  Ollama 未启动 → 检测仍可跑通，但决策只用规则知识库。"
    echo "   想用本地大模型请另开终端执行： OLLAMA_ORIGINS='*' ollama serve"
fi

# ---------- 1) FastAPI 指令总线 + 浏览器机械臂页面 ----------
if is_up "http://127.0.0.1:$API_PORT/api/health"; then
    echo "✅ FastAPI 已在运行 (http://127.0.0.1:$API_PORT)"
else
    echo "▶️  启动 FastAPI 指令总线 …"
    nohup "$PY" -m uvicorn api_server.arm_fastapi_server:app \
        --host 127.0.0.1 --port "$API_PORT" \
        > "$LOG_DIR/api_server.log" 2>&1 &
    for _ in $(seq 1 30); do
        is_up "http://127.0.0.1:$API_PORT/api/health" && break
        sleep 0.5
    done
    if is_up "http://127.0.0.1:$API_PORT/api/health"; then
        echo "✅ FastAPI 就绪 (http://127.0.0.1:${API_PORT}，日志 dataset_temp/api_server.log)"
    else
        echo "❌ FastAPI 启动失败，请看 dataset_temp/api_server.log"
        exit 1
    fi
fi

# ---------- 2) Streamlit 交互主界面 ----------
if is_up "http://127.0.0.1:$UI_PORT/_stcore/health"; then
    echo "✅ Streamlit 已在运行 (http://localhost:$UI_PORT)"
else
    echo "▶️  启动 Streamlit 交互界面 …"
    nohup "$PY" -m streamlit run ui_streamlit/streamlit_app.py \
        --server.port "$UI_PORT" --server.headless true \
        --browser.gatherUsageStats false \
        > "$LOG_DIR/streamlit.log" 2>&1 &
    for _ in $(seq 1 40); do
        is_up "http://127.0.0.1:$UI_PORT/_stcore/health" && break
        sleep 0.5
    done
    if is_up "http://127.0.0.1:$UI_PORT/_stcore/health"; then
        echo "✅ Streamlit 就绪 (http://localhost:${UI_PORT}，日志 dataset_temp/streamlit.log)"
    else
        echo "❌ Streamlit 启动失败，请看 dataset_temp/streamlit.log"
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo " 🏭 打开这两个页面即可使用"
echo "   ① 检测主界面   http://localhost:$UI_PORT"
echo "   ② 机械臂页面   http://127.0.0.1:$API_PORT/sim/"
echo ""
echo " 演示：主界面 →「🗂️ 示例图片」选 demo_defect_high.jpg →「🚀 开始检测」"
echo "       出现缺陷告警后点蓝色按钮 → 机械臂自动抓取分拣"
echo ""
echo " 停止：./start.sh --stop        查看状态：./start.sh --status"
echo "============================================================"

if [ "$OPEN_BROWSER" = "1" ]; then
    open "http://localhost:$UI_PORT" 2>/dev/null || true
    sleep 1
    open "http://127.0.0.1:$API_PORT/sim/" 2>/dev/null || true
fi
