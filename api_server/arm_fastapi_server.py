"""
FastAPI 中间通信服务：
- 维护一个全局"指令总线" bus，连接 Agent(Python) 与 浏览器虚拟机械臂(JS)；
- 浏览器每 500ms 轮询 GET /api/bus，按指令动作；动作完成后回调上报；
- Agent 通过 HTTP 设置拍照请求 / 目标点位，并等待浏览器完成；
- 同时静态托管 sim_web_static/，浏览器访问 /sim/ 即可打开虚拟机械臂页面。

启动：uvicorn api_server.arm_fastapi_server:app --port 8000 --reload
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import DATASET_TEMP_DIR, SIM_WEB_DIR, TOOL_Z_PICK, TOOL_Z_HOME

app = FastAPI(title="Arm Defect Agent - Bridge")

# 允许 Vite dev server (5173) 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 全局指令总线（进程内单例）----------
_bus_lock = threading.Lock()
_bus: dict = {
    "capture_request": False,     # Agent -> 浏览器：请拍一张照片
    "capture_ts": 0.0,            # 最近一次拍照请求时间（Streamlit 用）
    "target_pose": None,          # Agent -> 浏览器：[x, y, z]，None 表示不动
    "arm_arrived": True,          # 浏览器 -> Agent：已到达上一个目标点
    "latest_frame": None,         # 浏览器 -> Agent：最新上传帧的文件名
    "frame_ts": 0.0,              # 帧更新时间戳，Agent 用来感知"新帧到了"
    "running": True,              # 巡检开关
    "pick_task": None,            # Agent -> 浏览器：待执行分拣任务 {x,y,dest_x,dest_y,note,ts}
    "decision": None,             # Agent 决策结果（浏览器 HUD / Streamlit 同步展示）
    "detect": None,               # 最近一次 YOLO 结构化结果
    "decision_source": "",        # 决策来源：streamlit / agent / manual
    "arm_ready_ts": 0.0,          # 浏览器上报到位的时刻
}


class DispatchCmd(BaseModel):
    x: float
    y: float
    z: float | None = None
    note: str = ""


class PickCmd(BaseModel):
    """浏览器执行"抓取 → 投放到目标区 → 回零"完整动作序列。"""
    x: float
    y: float
    dest_x: float | None = None
    dest_y: float | None = None
    note: str = ""


class DecisionCmd(BaseModel):
    decision: dict = {}
    detect: dict = {}
    source: str = ""


@app.get("/api/health")
def health():
    """健康检查：Streamlit / 脚本启动前用它判断服务是否已就绪。"""
    return {"ok": True, "service": "arm-bridge", "ts": time.time()}


@app.get("/api/bus")
def read_bus():
    """浏览器轮询。只给浏览器看它需要的字段。"""
    with _bus_lock:
        return {
            "capture_request": _bus["capture_request"],
            "target_pose": _bus["target_pose"],
            "running": _bus["running"],
            "pick_task": _bus["pick_task"],
            "decision": _bus["decision"],
            "detect": _bus["detect"],
            "decision_source": _bus["decision_source"],
        }


@app.post("/api/request_capture")
def request_capture():
    """Agent 请求浏览器拍一张照片。"""
    with _bus_lock:
        _bus["capture_request"] = True
        _bus["capture_ts"] = time.time()
    return {"ok": True}


@app.post("/api/ack_capture")
def ack_capture():
    """浏览器已收到拍照请求，即将上传帧。"""
    with _bus_lock:
        _bus["capture_request"] = False
    return {"ok": True}


@app.post("/api/pick")
def pick(cmd: PickCmd):
    """
    下发一次完整分拣任务。浏览器轮询 /api/bus 拿到 pick_task 后执行
    "抓取 → 投放到目标区 → 回零"，结束后回调 /api/arm_ready。

    注意：即使浏览器页面还没打开，任务也会挂在总线上，等页面打开后自检执行。
    """
    with _bus_lock:
        _bus["pick_task"] = {
            "x": round(cmd.x, 4),
            "y": round(cmd.y, 4),
            "dest_x": round(cmd.dest_x, 4) if cmd.dest_x is not None else None,
            "dest_y": round(cmd.dest_y, 4) if cmd.dest_y is not None else None,
            "note": cmd.note,
            "ts": time.time(),
        }
        _bus["target_pose"] = [round(cmd.x, 4), round(cmd.y, 4), TOOL_Z_PICK]
        _bus["arm_arrived"] = False
    return {"ok": True, "pick_task": _bus["pick_task"]}


@app.get("/api/pick_task")
def get_pick_task():
    with _bus_lock:
        return {"pick_task": _bus["pick_task"], "arm_arrived": _bus["arm_arrived"]}


@app.post("/api/decision")
def post_decision(cmd: DecisionCmd):
    """Streamlit / Agent 把决策结果广播到总线，浏览器 HUD 与日志区同步展示。"""
    with _bus_lock:
        _bus["decision"] = cmd.decision
        _bus["detect"] = cmd.detect
        _bus["decision_source"] = cmd.source
    return {"ok": True}



@app.post("/api/upload_frame")
async def upload_frame(file: UploadFile = File(...)):
    """浏览器把 canvas 截图像素上传到 dataset_temp/。"""
    ts = time.time()
    fname = f"frame_{int(ts*1000)}.jpg"
    fpath = DATASET_TEMP_DIR / fname
    content = await file.read()
    fpath.write_bytes(content)
    with _bus_lock:
        _bus["latest_frame"] = fname
        _bus["frame_ts"] = ts
    return {"ok": True, "frame": fname}


@app.post("/api/arm_ready")
def arm_ready():
    """浏览器动画播完，机械臂到位。"""
    with _bus_lock:
        _bus["arm_arrived"] = True
        _bus["arm_ready_ts"] = time.time()
    return {"ok": True}


@app.post("/api/dispatch")
def dispatch(cmd: DispatchCmd):
    """Agent 下发目标点。"""
    z = cmd.z if cmd.z is not None else TOOL_Z_PICK
    with _bus_lock:
        _bus["target_pose"] = [round(cmd.x, 4), round(cmd.y, 4), round(z, 4)]
        _bus["arm_arrived"] = False   # 浏览器开始动，置 False
    return {"ok": True, "target": _bus["target_pose"]}


@app.post("/api/go_home")
def go_home():
    with _bus_lock:
        _bus["target_pose"] = [0.0, 0.0, TOOL_Z_HOME]
        _bus["arm_arrived"] = False
    return {"ok": True}


@app.get("/api/latest_frame")
def latest_frame():
    """Agent 查询最新一帧的本地路径。"""
    with _bus_lock:
        return {"frame": _bus["latest_frame"], "frame_ts": _bus["frame_ts"]}


@app.get("/api/frame/{name}")
def get_frame(name: str):
    p = DATASET_TEMP_DIR / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/state")
def state():
    with _bus_lock:
        return dict(_bus)


# ---------- 交互式 YOLO 检测接口（前端工作台用）----------
import cv2
import numpy as np
import base64


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    """前端上传一张图片，跑 YOLO，返回检测结果 + 画框后的 base64 图。

    同时用规则脑给出"这个缺陷该投到哪个区"，浏览器页面据此自动分拣。
    （需要大模型解读时走 Streamlit 界面，那边会调 Ollama 出决策报告。）
    """
    from yolo_module.yolo_defect_infer import detector
    from agent.decision import rule_decide

    content = await file.read()
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "图片解码失败")

    vis_path = DATASET_TEMP_DIR / f"vis_interactive_{int(time.time()*1000)}.jpg"
    result = detector.detect(img, save_vis_path=vis_path)
    decision = rule_decide(result)

    # 把画框图转 base64
    b64 = ""
    if vis_path.exists():
        b64 = base64.b64encode(vis_path.read_bytes()).decode("ascii")

    return {
        "has_defect": result["has_defect"],
        "defect_cls": result["defect_cls"],
        "conf": result["conf"],
        "bbox_xyxy": result["bbox_xyxy"],
        "world_xy": result["world_xy"],
        "all_detections": result["all_detections"],
        "vis_image": f"data:image/jpeg;base64,{b64}",
        "vis_path": str(vis_path),
        # 规则决策（浏览器端自动分拣用）
        "action": decision["action"],
        "dest": decision["dest"],
        "dest_label": decision["dest_label"],
        "reason": decision["reason"],
    }


@app.post("/api/set_running")
def set_running(running: int):
    with _bus_lock:
        _bus["running"] = bool(running)
    return {"ok": True, "running": _bus["running"]}


# ---------- 静态托管虚拟机械臂页面 ----------
SIM_WEB_DIR.mkdir(exist_ok=True)
app.mount("/sim", StaticFiles(directory=str(SIM_WEB_DIR), html=True), name="sim")


@app.get("/")
def root():
    return JSONResponse({
        "msg": "Arm Defect Agent Bridge",
        "sim_page": "/sim/",
        "docs": "/docs",
    })
