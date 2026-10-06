"""
YOLOv8 钢材缺陷检测封装。
- 加载 best.pt 权重做推理；
- OpenCV 做 ROI 截取 + HSV 阈值过滤背景；
- 像素 bbox -> 仿真工作台世界坐标（线性映射，仿真环境下的"虚拟标定"）；
- 把检测框画在图上另存一份可视化结果，供 Streamlit 展示。

如果本机没装 ultralytics，自动降级为"模拟检测器"，项目仍能跑通闭环演示。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import (
    YOLO_BEST_PT,
    CAM_IMG_W, CAM_IMG_H,
    TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX,
    YOLO_CONF_THRES,
)


# ---------------- HSV 背景过滤参数（可按仿真背景现场调）----------------
# 仿真工作台默认是浅灰/银色背景，这里把接近背景的像素压暗，突出缺陷。
# 真实项目里这些值用黑马课程里学过的 HSV 阈值滑块现场标定。
HSV_LOWER = np.array([0, 0, 0])
HSV_UPPER = np.array([180, 60, 255])   # 低饱和度视为背景

# 送进模型的置信度下限：要比业务阈值低，否则"0.35~0.6 之间需要复检"这类中间态
# 会被模型直接丢掉，业务层根本拿不到。真正的判定阈值由调用方传入。
MODEL_CONF_FLOOR = 0.05


class DefectDetector:
    _instance: Optional["DefectDetector"] = None
    _model = None
    _model_loaded = False
    _use_mock = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """尝试加载 ultralytics YOLO；失败则降级 mock。"""
        try:
            from ultralytics import YOLO  # noqa
            if not YOLO_BEST_PT.exists():
                raise FileNotFoundError(f"权重不存在: {YOLO_BEST_PT}")
            self._model = YOLO(str(YOLO_BEST_PT))
            # 热身一次（首次推理会慢）
            dummy = np.zeros((CAM_IMG_H, CAM_IMG_W, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False)
            self._model_loaded = True
            print(f"[YOLO] 模型加载成功: {YOLO_BEST_PT.name}, 类别={self._model.names}")
        except Exception as e:
            self._use_mock = True
            print(f"[YOLO] ⚠️ 未能加载真实模型({e})，已降级为模拟检测器（项目仍可跑通闭环）")

    # ---------------- OpenCV 预处理 ----------------
    @staticmethod
    def preprocess(img_bgr: np.ndarray) -> np.ndarray:
        """ROI 截取 + HSV 背景抑制。这里整张图即 ROI（虚拟相机已对准工件）。"""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        # 背景区域变白，缺陷保留原色
        bg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        return bg

    # ---------------- 坐标映射 ----------------
    @staticmethod
    def pixel_to_world(cx_px: float, cy_px: float) -> tuple[float, float]:
        """
        像素坐标(检测框中心) -> 仿真工作台世界坐标 (x, y)。
        仿真环境下做线性投影映射；真实产线需要相机内外参标定。
        """
        x = TABLE_X_MIN + (cx_px / CAM_IMG_W) * (TABLE_X_MAX - TABLE_X_MIN)
        y = TABLE_Y_MIN + (cy_px / CAM_IMG_H) * (TABLE_Y_MAX - TABLE_Y_MIN)
        return round(float(x), 4), round(float(y), 4)

    # ---------------- 推理主入口 ----------------
    def detect(self, img_bgr: np.ndarray, save_vis_path: Optional[Path] = None,
               conf_thres: float = YOLO_CONF_THRES) -> dict:
        """
        输入 BGR 图像（来自浏览器虚拟相机上传），返回结构化检测结果。
        conf_thres: 判定"有缺陷"的置信度阈值，界面可实时调整。
        返回字段：
          has_defect, defect_cls, conf, bbox_xyxy, world_xy, all_detections, vis_path
        """
        pre = self.preprocess(img_bgr)

        if self._use_mock:
            dets = self._mock_detections(img_bgr)
        else:
            dets = self._real_detect(pre)

        # 取置信度最高的一个缺陷作为当前帧结论
        if dets:
            best = max(dets, key=lambda d: d["conf"])
            cx = (best["bbox_xyxy"][0] + best["bbox_xyxy"][2]) / 2
            cy = (best["bbox_xyxy"][1] + best["bbox_xyxy"][3]) / 2
            wx, wy = self.pixel_to_world(cx, cy)
            result = {
                "has_defect": best["conf"] >= conf_thres,
                "defect_cls": best["cls"],
                "conf": round(best["conf"], 3),
                "bbox_xyxy": best["bbox_xyxy"],
                "world_xy": [wx, wy],
                "all_detections": dets,
                "conf_thres": conf_thres,
            }
        else:
            result = {
                "has_defect": False, "defect_cls": "", "conf": 0.0,
                "bbox_xyxy": [], "world_xy": None, "all_detections": [],
                "conf_thres": conf_thres,
            }

        # 画框可视化
        if save_vis_path is not None:
            vis = img_bgr.copy()
            for d in dets:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                color = (0, 0, 255) if d["conf"] >= conf_thres else (0, 200, 200)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{d['cls']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imwrite(str(save_vis_path), vis)
            result["vis_path"] = str(save_vis_path)
        return result

    # ---------------- 真实 YOLO 推理 ----------------
    def _real_detect(self, img_bgr: np.ndarray) -> list[dict]:
        # 注意：模型侧用很低的 floor，把"低置信度候选"也拿回来，
        # 具体判不判缺陷由 detect(conf_thres=...) 决定。
        preds = self._model.predict(img_bgr, conf=MODEL_CONF_FLOOR, verbose=False)
        dets = []
        if not preds:
            return dets
        boxes = preds[0].boxes
        names = self._model.names
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            dets.append({"cls": names.get(cls_id, str(cls_id)), "conf": conf, "bbox_xyxy": xyxy})
        return dets

    # ---------------- 模拟检测（兜底）----------------
    def _mock_detections(self, img_bgr: np.ndarray) -> list[dict]:
        """没有真实模型时，随机撒一个缺陷框，让闭环能演示。"""
        h, w = img_bgr.shape[:2]
        rng = np.random.default_rng(int(time.time()) % 10000)
        if rng.random() < 0.55:  # 55% 概率检出缺陷
            bw, bh = rng.integers(40, 90), rng.integers(20, 50)
            x1 = int(rng.integers(20, w - bw - 20))
            y1 = int(rng.integers(20, h - bh - 20))
            cls = rng.choice(["scratch", "rust", "inclusion"])
            conf = float(rng.uniform(0.45, 0.92))
            return [{"cls": cls, "conf": conf, "bbox_xyxy": [x1, y1, x1 + bw, y1 + bh]}]
        return []


# 模块级单例
detector = DefectDetector()
