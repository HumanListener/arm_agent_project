"""
全局配置：Ollama 地址、FastAPI 端口、YOLO 权重路径、像素→工作台坐标映射。
所有模块统一从这里读配置，避免硬编码。
"""
import os
from pathlib import Path

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS_DIR = PROJECT_ROOT / "yolo_module" / "weights"
YOLO_BEST_PT = YOLO_WEIGHTS_DIR / "best.pt"
DATASET_TEMP_DIR = PROJECT_ROOT / "dataset_temp"
DB_DIR = PROJECT_ROOT / "db"
SIM_WEB_DIR = PROJECT_ROOT / "sim_web_static"

DATASET_TEMP_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# ---------- Ollama 本地大模型 ----------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# 本机 `ollama list` 已有的模型：deepseek-r1:8b / deepseek-v4-flash:cloud / deepseek-v4-pro:cloud
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_TIMEOUT = 180  # 秒（8B 模型在 CPU 上单次决策实测约 70s，留足余量）
OLLAMA_FALLBACK_MODELS = ["deepseek-r1:8b", "qwen2.5:7b", "deepseek-r1:7b", "llama3.1:8b"]

# ---------- FastAPI 中间通信服务 ----------
API_HOST = "127.0.0.1"
API_PORT = 8000
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# ---------- Streamlit 控制面板 ----------
STREAMLIT_PORT = 8501

# ---------- 虚拟相机 / 工作台尺寸（像素与仿真世界的映射）----------
# 虚拟相机看到的画面分辨率
CAM_IMG_W = 640
CAM_IMG_H = 480
# 工作台在仿真世界里的物理范围（米）。YOLO 检测到的像素坐标按线性映射落到这个平面。
TABLE_X_MIN, TABLE_X_MAX = -0.30, 0.30   # 机械臂正前方 X 方向物理范围
TABLE_Y_MIN, TABLE_Y_MAX = -0.20, 0.20   # 横向 Y 方向物理范围
TOOL_Z_PICK = 0.05                        # 抓取/靠近工件时末端 Z 高度
TOOL_Z_HOME = 0.15                        # HOME 点高度

# ---------- 巡检点位（工作台上方 3x3 网格，覆盖一块钢板）----------
INSPECT_GRID = [
    (-0.20, -0.12), (-0.00, -0.12), (0.20, -0.12),
    (-0.20,  0.00), ( 0.00,  0.00), (0.20,  0.00),
    (-0.20,  0.12), ( 0.00,  0.12), (0.20,  0.12),
]

# ---------- YOLO 阈值 ----------
YOLO_CONF_THRES = 0.25     # 低于此置信度视为"看不清"，转人工复检（不是直接丢帧）
YOLO_CONF_MIN = 0.05       # 界面滑块下限
YOLO_CONF_MAX = 0.90       # 界面滑块上限
DEFECT_HIGH_CONF = 0.60    # 高于此置信度视为"明确缺陷"，可直接分拣

# ---------- 分拣目标区（与 sim_web_static/index.html 里的三个圆盘一致）----------
# 这三个坐标必须是机械臂 IK 真能到达的点（底座在世界系 (0,0.06,-0.25)，总臂展≈0.36m）：
#   实测可达区域大致是以底座为圆心、半径 0.10~0.34m 的扇形，且 z 不能太靠后（z<-0.2 基本够不到）。
# 改坐标前请先跑 scratch/ik_workspace.mjs 验证可达性。
DEST_BINS = {
    "ng":     {"label": "废品区",   "xy": [ 0.20, 0.00]},   # 严重缺陷 → 报废
    "rework": {"label": "返修区",   "xy": [ 0.00, 0.05]},   # 轻微缺陷 → 打磨返修
    "review": {"label": "复检区",   "xy": [-0.20, 0.00]},   # 不确定   → 人工复检
}

# ---------- MCP 机械臂工具服务 ----------
ARM_MCP_SERVER = PROJECT_ROOT / "mcp_service" / "mcp_arm_tools.py"
ARM_TOOL_TIMEOUT = 25      # 单个 MCP 工具调用超时（秒）
ARM_SETTLE_SEC = 0.8       # 虚拟机械臂动作后的稳定时间

# ---------- 自动巡检的决策方式 ----------
# True  = 每条检测结果都问本地大模型（更"智能"，但 8B CPU 推理约 60~90s/点，9 个点会很慢）
# False = 只用规则知识库（秒级，演示更流畅）；Streamlit 界面上可以按需临时开关
USE_LLM = os.getenv("USE_LLM", "0") == "1"

# ---------- SQLite 日志库 ----------
DB_PATH = DB_DIR / "inspection.db"
