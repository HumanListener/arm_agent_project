"""
生成演示用样例图片（samples/）。

为什么需要它：
    现场演示/答辩时，如果手边没有真实缺陷图，整条"感知 → 决策 → 行动"链路就演示不出来。
    这个脚本用 OpenCV 合成三种典型钢板表面图，并用当前 best.pt 实测出各自的检测置信度，
    保证"高置信缺陷"那张确实能触发分拣动作。

用法：
    python tools/make_samples.py            # 生成到 samples/
    python tools/make_samples.py --check    # 只跑一遍模型确认检测结果
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES_DIR = ROOT / "samples"


def _steel_bg(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    """合成 NEU-DET 风格的中灰钢板背景（带轻微轧制纹理）。"""
    rng = np.random.default_rng(seed)
    gray = rng.normal(140, 8, (h, w)).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_clean(seed: int = 11) -> np.ndarray:
    """无缺陷：直接放行。"""
    return _steel_bg(seed)


def make_defect_high(seed: int = 3) -> np.ndarray:
    """高置信缺陷：一块深色夹杂斑块（实测 inclusion ≈ 0.90，触发分拣）。"""
    img = _steel_bg(seed)
    cv2.ellipse(img, (330, 250), (130, 80), 20, 0, 360, (78, 76, 74), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def make_defect_mid(seed: int = 7) -> np.ndarray:
    """低置信缺陷：细微划痕（实测 rolled-in_scale ≈ 0.24，转人工复检）。"""
    rng = np.random.default_rng(seed)
    img = _steel_bg(seed)
    cv2.line(img, (150, 150), (470, 300), (58, 58, 62), 6, cv2.LINE_AA)
    cv2.line(img, (150, 152), (470, 302), (95, 95, 100), 2, cv2.LINE_AA)
    for _ in range(25):
        x, y = int(rng.integers(60, 580)), int(rng.integers(60, 420))
        cv2.circle(img, (x, y), int(rng.integers(2, 5)), (72, 72, 76), -1)
    return img


def main(check_only: bool = False):
    SAMPLES_DIR.mkdir(exist_ok=True)
    makers = {
        "demo_clean.jpg": make_clean,
        "demo_defect_high.jpg": make_defect_high,
        "demo_defect.jpg": make_defect_mid,
    }
    if not check_only:
        for name, fn in makers.items():
            cv2.imwrite(str(SAMPLES_DIR / name), fn(), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"[生成] samples/{name}")

    # 用真实模型确认每张图的检测结果
    try:
        from yolo_module.yolo_defect_infer import detector
    except Exception as e:  # noqa: BLE001
        print(f"[跳过检测] 无法加载检测器: {e}")
        return
    print("\n=== 实测检测结果 ===")
    for name in makers:
        img = cv2.imread(str(SAMPLES_DIR / name))
        if img is None:
            continue
        r = detector.detect(img)
        print(f"samples/{name}: has_defect={r['has_defect']} cls={r['defect_cls'] or '-'} "
              f"conf={r['conf']} world_xy={r['world_xy']}")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
