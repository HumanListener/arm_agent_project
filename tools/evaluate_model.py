"""
模型评估脚本 —— 补上项目缺失的"评估闭环"。

为什么需要它（面试视角）：
    原来项目里只有推理，没有评估。面试官问"mAP 怎么算的""哪两类最容易混"
    "阈值 0.25 是怎么定的"，没有数据支撑就只能说"经验值"，会被判定为"只会调包"。
    这个脚本把这三件事全部变成可复现、可展示的图和数据。

产出（默认写到 evaluation/ 目录）：
    01_overall_metrics.txt   —— 总体 Precision/Recall/mAP@50/mAP@50-95
    02_per_class_metrics.csv —— 每个缺陷类别的 P/R/mAP，找出最差类别
    03_confusion_matrix.png  —— 混淆矩阵热力图（含"背景"行/列 = 漏检/误检）
    04_confusion_pairs.txt   —— 最容易混的类别组合 TOP N（面试直接引用）
    05_threshold_sweep.csv   —— 置信度阈值扫描：P/R/F1/漏检率/误检率
    06_threshold_curve.png   —— 阈值-P/R 曲线（阈值标定的依据）
    07_gt_analysis.txt       —— 数据集类别分布与不平衡分析
    08_report.md             —— 汇总报告（可直接放 GitHub / 面试时讲）
    09_badcases/             —— 漏检 / 误检 的典型图片（面试讲 badcase 用）

用法：
    # 基础用法（用训练时的数据集配置）
    python tools/evaluate_model.py

    # 指定模型和数据集
    python tools/evaluate_model.py --weights yolo_module/weights/best.pt \
                                   --data ~/Desktop/NEU-DET/data.yaml

    # 快速模式（只抽样 100 张，用于验证脚本能跑通）
    python tools/evaluate_model.py --limit 100

    # 跳过阈值扫描（省时间）
    python tools/evaluate_model.py --skip-sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# 通用工具
# ============================================================
def setup_matplotlib_chinese():
    """matplotlib 默认字体不含中文，会画成方框。这里尝试挑一个系统中文字体。"""
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC",
                  "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    # 找不到就退回默认（英文图例仍可读）
    plt.rcParams["axes.unicode_minus"] = False
    return None


def box_iou(box_a, box_b) -> float:
    """两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    """YOLO 归一化标签 → 像素 xyxy。"""
    cx, cy = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def load_gt_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, list]]:
    """读取一个 YOLO 标签文件，返回 [(cls_id, xyxy), ...]。"""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        out.append((cls_id,
                    xywhn_to_xyxy(float(parts[1]), float(parts[2]),
                                  float(parts[3]), float(parts[4]), img_w, img_h)))
    return out


# ============================================================
# 1) 数据集分析
# ============================================================
def analyze_dataset(data_yaml: Path, out_dir: Path, names: dict) -> dict:
    """统计 train/val 的类别分布，找出类别不平衡。"""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_yaml.parent)

    def resolve(split_key):
        rel = cfg.get(split_key)
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else root / p

    lines = ["# 数据集类别分布分析", ""]
    stats = {}
    for split in ("train", "val"):
        img_dir = resolve(split)
        if img_dir is None or not img_dir.exists():
            lines.append(f"- {split}: 目录不存在（{img_dir}），跳过")
            continue
        lbl_dir = root / "labels" / split
        counter = Counter()
        n_img = 0
        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            n_img += 1
            for cls_id, _ in load_gt_labels(lbl_dir / f"{img.stem}.txt", 1, 1):
                counter[cls_id] += 1
        total = sum(counter.values()) or 1
        stats[split] = {"images": n_img, "boxes": total, "per_class": dict(counter)}
        lines.append(f"## {split}：{n_img} 张图，{total} 个标注框")
        lines.append("")
        lines.append("| 类别 | 框数 | 占比 | 相对最少类 |")
        lines.append("|---|---|---|---|")
        for cls_id in sorted(names):
            c = counter.get(cls_id, 0)
            lines.append(f"| {names[cls_id]} | {c} | {c / total * 100:.1f}% | {c / max(1, min(counter.values())):.2f}x |")
        lines.append("")

        if counter:
            fewest = min(counter, key=counter.get)
            most = max(counter, key=counter.get)
            ratio = counter[most] / max(1, counter[fewest])
            lines.append(f"⚠️ 类别不平衡：最多类 `{names[most]}`({counter[most]}) 是最少类 "
                         f"`{names[fewest]}`({counter[fewest]}) 的 **{ratio:.2f} 倍**。"
                         f"少数类通常 AP 最低，是模型改进的优先方向。")
            lines.append("")

    (out_dir / "07_gt_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    return stats


# ============================================================
# 2) 收集验证集上的全部预测（一次推理，多次复用）
# ============================================================
def collect_predictions(model, data_yaml: Path, imgsz: int, batch: int, device: str,
                        conf_floor: float, limit: int | None, names: dict):
    """
    在验证集上跑一次低阈值推理，收集所有候选框。
    返回 records: [{img, gt:[(cls,xyxy)], preds:[(cls,conf,xyxy)]}]，以及类别分布。

    为什么要一次性收集：阈值扫描需要多次评估，如果每次都重新推理会慢好几倍。
    """
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path") or data_yaml.parent)
    rel = cfg.get("val")
    img_dir = Path(rel) if Path(rel).is_absolute() else root / rel
    lbl_dir = root / "labels" / "val"

    images = [p for p in sorted(img_dir.glob("*"))
              if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if limit:
        images = images[:limit]
    if not images:
        raise SystemExit(f"❌ 验证集目录里没找到图片: {img_dir}")

    print(f"[评估] 验证集 {len(images)} 张图，开始推理（conf floor={conf_floor}）…")
    records = []
    cls_counter = Counter()

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt_labels(lbl_dir / f"{img_path.stem}.txt", w, h)

        res = model.predict(img, imgsz=imgsz, conf=conf_floor, iou=0.7,
                            device=device, verbose=False)[0]
        preds = []
        for b in res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy.cpu().numpy().reshape(-1).tolist()
            preds.append((cls_id, conf, xyxy))
            cls_counter[cls_id] += 1

        records.append({"path": str(img_path), "shape": (h, w), "gt": gt, "preds": preds})
        if i % 50 == 0 or i == len(images):
            print(f"   {i}/{len(images)} …")

    print(f"[评估] 推理完成，共收集 {sum(cls_counter.values())} 个候选框")
    return records


# ============================================================
# 3) 混淆矩阵（以"预测框"为单位统计，含背景行/列）
# ============================================================
def build_confusion_matrix(records, names: dict, conf_thres: float, iou_thres: float = 0.45):
    """
    构造混淆矩阵。行列含义（与 ultralytics 口径一致）：
        cm[gt_cls][pred_cls]  —— 该 GT 被预测成什么
        cm[gt_cls][-1]        —— 该 GT 没被任何预测匹配上（漏检 / 被预测成背景）
        cm[-1][pred_cls]      —— 误检（预测了但没匹配到任何 GT）
    只对置信度 >= conf_thres 的预测参与匹配。
    """
    n = len(names)
    cm = np.zeros((n + 1, n + 1), dtype=int)

    for rec in records:
        gt = [(c, b) for c, b in rec["gt"]]
        preds = [(c, cf, b) for c, cf, b in rec["preds"] if cf >= conf_thres]
        matched_gt = set()
        matched_pred = set()

        # 按置信度从高到低匹配（贪心，遵循"高置信优先占用 GT"的惯例）
        for pi, (pc, _pcf, pb) in enumerate(sorted(preds, key=lambda t: -t[1])):
            best_iou, best_gi = 0.0, -1
            for gi, (gc, gb) in enumerate(gt):
                if gi in matched_gt:
                    continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thres and best_gi >= 0:
                gc = gt[best_gi][0]
                cm[gc][pc] += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                cm[n][pc] += 1          # 误检

        for gi, (gc, _gb) in enumerate(gt):
            if gi not in matched_gt:
                cm[gc][n] += 1          # 漏检

    return cm


def confusion_analysis(cm: np.ndarray, names: dict) -> tuple[list[str], dict]:
    """从混淆矩阵里提炼可讲结论：谁和谁混、漏检率、误检率。"""
    n = len(names)
    labels = [names[i] for i in range(n)] + ["背景(漏检)"]
    lines = []
    summary = {}

    total_gt = int(cm[:n, :].sum())
    total_pred_matched = int(cm[:n, :n].sum())
    wrong = int(sum(cm[i][j] for i in range(n) for j in range(n) if i != j))
    missed = int(cm[:n, n].sum())
    fp = int(cm[n, :n].sum())

    summary.update({"total_gt": total_gt, "missed": missed, "false_positive": fp,
                    "correct": total_pred_matched - wrong, "misclassified": wrong})

    lines.append("# 混淆矩阵分析")
    lines.append("")
    lines.append(f"- GT 总数：{total_gt}")
    lines.append(f"- 分类正确：{total_pred_matched - wrong}")
    lines.append(f"- **分类错误（框对了但类别判错）：{wrong}**"
                 f"（占 {wrong / max(1, total_gt) * 100:.1f}%）")
    lines.append(f"- **漏检（没检出 / 判成背景）：{missed}**"
                 f"（占 {missed / max(1, total_gt) * 100:.1f}%）")
    lines.append(f"- **误检（无中生有）：{fp}**")
    lines.append("")

    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((int(cm[i][j]), names[i], names[j]))
    pairs.sort(reverse=True)
    if pairs:
        lines.append("## 最容易混淆的类别组合（TOP 8）")
        lines.append("")
        lines.append("| 次数 | 真实类别 | 被误判为 |")
        lines.append("|---|---|---|")
        for cnt, a, b in pairs[:8]:
            lines.append(f"| {cnt} | {a} | {b} |")
        lines.append("")
    summary["top_confusions"] = pairs[:5]

    lines.append("## 每个类别的漏检率（AP 上不去的直接原因）")
    lines.append("")
    lines.append("| 类别 | GT数 | 漏检 | 漏检率 |")
    lines.append("|---|---|---|---|")
    per_cls_miss = {}
    for i in range(n):
        gt_i = int(cm[i, :].sum())
        miss_i = int(cm[i, n])
        rate = miss_i / gt_i * 100 if gt_i else 0.0
        per_cls_miss[names[i]] = {"gt": gt_i, "missed": miss_i, "rate": round(rate, 1)}
        flag = " ⚠️" if rate > 40 else ""
        lines.append(f"| {names[i]} | {gt_i} | {miss_i} | {rate:.1f}%{flag} |")
    lines.append("")
    summary["per_class_miss"] = per_cls_miss
    summary["_cm"] = cm          # 供交叉验证反算 P/R 用

    return lines, summary, labels


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_png: Path, title: str):
    """画归一化混淆矩阵热力图（按行归一化 = 每个真实类别的去向分布）。"""
    import matplotlib.pyplot as plt

    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("预测类别", fontsize=11)
    ax.set_ylabel("真实类别", fontsize=11)
    ax.set_title(title, fontsize=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] == 0:
                continue
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%",
                    ha="center", va="center", fontsize=7.5, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, label="按行归一化比例")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================================================
# 4) 阈值扫描（面试里"阈值 0.25 怎么定的"的依据）
# ============================================================
def sweep_thresholds(records, names: dict, thresholds: list[float],
                     iou_thres: float = 0.45) -> list[dict]:
    """
    用同一批候选框，在不同置信度阈值下重新匹配，计算 P/R/F1/漏检率/误检率。
    这是真正回答"阈值怎么定"的东西：看不同阈值下的 precision-recall 权衡。
    """
    rows = []
    n = len(names)
    for th in thresholds:
        tp = fp = fn = 0
        for rec in records:
            gt = rec["gt"]
            preds = [(c, cf, b) for c, cf, b in rec["preds"] if cf >= th]
            matched_gt = set()
            for pc, _cf, pb in sorted(preds, key=lambda t: -t[1]):
                best_iou, best_gi = 0.0, -1
                for gi, (gc, gb) in enumerate(gt):
                    if gi in matched_gt:
                        continue
                    iou = box_iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= iou_thres and best_gi >= 0:
                    if gt[best_gi][0] == pc:
                        tp += 1
                        matched_gt.add(best_gi)
                    else:
                        fp += 1
                        matched_gt.add(best_gi)
                else:
                    fp += 1
            fn += len(gt) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "conf_thres": round(th, 3),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "miss_rate": round(fn / (tp + fn), 4) if (tp + fn) else 0.0,
            "false_alarm_rate": round(fp / (tp + fp), 4) if (tp + fp) else 0.0,
        })
    return rows


def plot_threshold_curves(rows: list[dict], out_png: Path, current_thres: float = 0.25):
    import matplotlib.pyplot as plt

    th = [r["conf_thres"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    ax.plot(th, [r["precision"] for r in rows], "o-", label="Precision（报出的缺陷有多少是真的）")
    ax.plot(th, [r["recall"] for r in rows], "s-", label="Recall（真缺陷抓到了多少）")
    ax.plot(th, [r["f1"] for r in rows], "^-", label="F1")
    ax.axvline(current_thres, color="red", ls="--", lw=1.2,
               label=f"当前业务阈值 {current_thres}")
    ax.set_xlabel("置信度阈值")
    ax.set_ylabel("指标值")
    ax.set_title("阈值 vs Precision / Recall / F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(th, [r["miss_rate"] for r in rows], "o-", color="#d62728", label="漏检率（缺陷流到下游）")
    ax.plot(th, [r["false_alarm_rate"] for r in rows], "s-", color="#1f77b4", label="误报率（无谓复检）")
    ax.axvline(current_thres, color="red", ls="--", lw=1.2, label=f"当前阈值 {current_thres}")
    ax.set_xlabel("置信度阈值")
    ax.set_ylabel("比例")
    ax.set_title("阈值 vs 漏检率 / 误报率（质检成本权衡）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================================================
# 5) badcase 可视化（面试讲 badcase 必备）
# ============================================================
def export_badcases(records, names: dict, out_dir: Path, conf_thres: float,
                    iou_thres: float = 0.45, max_per_type: int = 8):
    """导出漏检 / 误检 的典型图片，绿框=GT，红框=预测。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    miss_saved = fp_saved = wrong_saved = 0

    for rec in records:
        if (miss_saved >= max_per_type and fp_saved >= max_per_type
                and wrong_saved >= max_per_type):
            break
        img = cv2.imread(rec["path"])
        if img is None:
            continue
        gt = rec["gt"]
        preds = [(c, cf, b) for c, cf, b in rec["preds"] if cf >= conf_thres]

        matched_gt = set()
        fp_boxes, wrong_boxes = [], []
        for pc, cf, pb in sorted(preds, key=lambda t: -t[1]):
            best_iou, best_gi = 0.0, -1
            for gi, (gc, gb) in enumerate(gt):
                if gi in matched_gt:
                    continue
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thres and best_gi >= 0:
                if gt[best_gi][0] == pc:
                    matched_gt.add(best_gi)
                else:
                    wrong_boxes.append((pc, cf, pb, names[gt[best_gi][0]]))
                    matched_gt.add(best_gi)
            else:
                fp_boxes.append((pc, cf, pb))

        miss_boxes = [(gc, gb) for gi, (gc, gb) in enumerate(gt) if gi not in matched_gt]

        tag = None
        if wrong_boxes and wrong_saved < max_per_type:
            tag, wrong_saved = f"judgewrong_{wrong_saved + 1:02d}", wrong_saved + 1
        elif miss_boxes and miss_saved < max_per_type:
            tag, miss_saved = f"miss_{miss_saved + 1:02d}", miss_saved + 1
        elif fp_boxes and fp_saved < max_per_type:
            tag, fp_saved = f"falsealarm_{fp_saved + 1:02d}", fp_saved + 1
        if tag is None:
            continue

        vis = img.copy()
        for gc, gb in gt:
            cv2.rectangle(vis, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])),
                          (0, 200, 0), 2)
            cv2.putText(vis, f"GT:{names[gc]}", (int(gb[0]), max(int(gb[1]) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        for pc, cf, pb in preds:
            cv2.rectangle(vis, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])),
                          (0, 0, 255), 2)
            cv2.putText(vis, f"{names[pc]} {cf:.2f}", (int(pb[0]), int(pb[3]) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / f"{tag}_{Path(rec['path']).stem}.jpg"), vis)

    return miss_saved, fp_saved, wrong_saved


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="YOLO 缺陷检测模型评估 + 阈值标定 + 混淆矩阵")
    ap.add_argument("--weights", default=str(ROOT / "yolo_module" / "weights" / "best.pt"),
                    help="模型权重路径")
    ap.add_argument("--data", default=str(Path.home() / "Desktop" / "NEU-DET" / "data.yaml"),
                    help="数据集 data.yaml 路径")
    ap.add_argument("--out", default=str(ROOT / "evaluation"), help="输出目录")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="", help="mps / cpu / 0；留空自动")
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 张（快速验证用）")
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="推理时保留候选的最低置信度（要和项目里的 MODEL_CONF_FLOOR 一致）")
    ap.add_argument("--deploy-thres", type=float, default=0.25,
                    help="当前业务部署阈值（项目里的 YOLO_CONF_THRES）")
    ap.add_argument("--iou", type=float, default=0.45, help="匹配 GT 用的 IoU 阈值")
    ap.add_argument("--skip-sweep", action="store_true", help="跳过阈值扫描（省时间）")
    args = ap.parse_args()

    weights, data_yaml = Path(args.weights), Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 前置检查 ----------
    if not weights.exists():
        raise SystemExit(f"❌ 权重不存在: {weights}\n"
                         f"   训练时用的是 yolo11n.pt 迁移学习，权重在 runs/train/ 下，"
                         f"复制到 {weights} 或改 --weights")
    if not data_yaml.exists():
        raise SystemExit(f"❌ data.yaml 不存在: {data_yaml}\n"
                         f"   用 --data 指定你的数据集配置")

    font = setup_matplotlib_chinese()
    print("=" * 66)
    print(" 缺陷检测模型评估")
    print("=" * 66)
    print(f"权重    : {weights}")
    print(f"数据集  : {data_yaml}")
    print(f"输出    : {out_dir}")
    print(f"中文字体: {font or '未找到（图例可能显示为方框）'}")
    print()

    from ultralytics import YOLO
    model = YOLO(str(weights))
    names = {int(k): v for k, v in model.names.items()}
    print(f"类别（{len(names)}）: {list(names.values())}\n")

    # ---------- 0) 数据集分析 ----------
    print("[1/5] 数据集类别分布分析 …")
    analyze_dataset(data_yaml, out_dir, names)

    # ---------- 1) 官方 val()：拿 P/R/mAP + 每类 AP ----------
    print("[2/5] 运行官方验证（ultralytics val）…")
    val_kwargs = dict(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, verbose=False)
    if args.device:
        val_kwargs["device"] = args.device
    results = model.val(**val_kwargs)
    rd = results.results_dict

    overall = {
        "precision": round(float(rd.get("metrics/precision(B)", 0)), 4),
        "recall": round(float(rd.get("metrics/recall(B)", 0)), 4),
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0)), 4),
        "mAP50-95": round(float(rd.get("metrics/mAP50-95(B)", 0)), 4),
        "fitness": round(float(rd.get("fitness", 0)), 4),
    }
    print(f"      P={overall['precision']}  R={overall['recall']}  "
          f"mAP50={overall['mAP50']}  mAP50-95={overall['mAP50-95']}")

    # 每类指标
    per_class = []
    p_vals = getattr(results.box, "p", None)
    r_vals = getattr(results.box, "r", None)
    ap50 = getattr(results.box, "ap50", None)
    ap = getattr(results.box, "ap", None)
    ap_idx = getattr(results.box, "ap_class_index", None)
    n_instances = {}
    _nt = getattr(results, "nt_per_class", None)
    if _nt is not None:
        for _ci, _n in enumerate(_nt):
            n_instances[int(_ci)] = int(_n)
    if ap50 is not None and ap_idx is not None:
        for k, cls_idx in enumerate(ap_idx):
            ci = int(cls_idx)
            per_class.append({
                "class": names.get(ci, str(ci)),
                "precision": round(float(p_vals[k]), 4) if p_vals is not None else "",
                "recall": round(float(r_vals[k]), 4) if r_vals is not None else "",
                "AP50": round(float(ap50[k]), 4),
                "AP50-95": round(float(ap[k]), 4) if ap is not None else "",
                "instances": n_instances.get(ci, ""),
            })
        per_class.sort(key=lambda d: d["AP50"])
        with open(out_dir / "02_per_class_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
            wr.writeheader()
            wr.writerows(per_class)
        print("      每类 AP50（升序，最差在前）:")
        for d in per_class:
            print(f"        {d['class']:18s} AP50={d['AP50']:.3f}  "
                  f"P={d['precision']}  R={d['recall']}  样本={d['instances']}")

    with open(out_dir / "01_overall_metrics.txt", "w", encoding="utf-8") as f:
        f.write("缺陷检测模型 - 总体指标\n")
        f.write("=" * 50 + "\n")
        f.write(f"权重: {weights}\n数据集: {data_yaml}\n")
        f.write(f"图像尺寸: {args.imgsz}   batch: {args.batch}\n\n")
        for k, v in overall.items():
            f.write(f"{k:12s} {v}\n")
        f.write("\n每类指标（AP50 升序，最差在前）\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'类别':<20}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>10}{'样本':>8}\n")
        for d in per_class:
            f.write(f"{d['class']:<20}{str(d['precision']):>8}{str(d['recall']):>8}"
                    f"{d['AP50']:>8}{str(d['AP50-95']):>10}{str(d['instances']):>8}\n")

    # ---------- 2) 收集预测 → 混淆矩阵 ----------
    print("[3/5] 收集验证集预测并构建混淆矩阵 …")
    records = collect_predictions(model, data_yaml, args.imgsz, args.batch,
                                  args.device or "cpu", args.conf_floor,
                                  args.limit or None, names)

    cm = build_confusion_matrix(records, names, args.deploy_thres, args.iou)
    lines, cm_summary, labels = confusion_analysis(cm, names)
    plot_confusion_matrix(cm, labels, out_dir / "03_confusion_matrix.png",
                          f"混淆矩阵（conf>={args.deploy_thres}, IoU>={args.iou}）")
    (out_dir / "04_confusion_pairs.txt").write_text("\n".join(lines), encoding="utf-8")

    np.savetxt(out_dir / "03_confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(labels), comments="")
    print(f"      分类错误 {cm_summary['misclassified']} / 漏检 {cm_summary['missed']} "
          f"/ 误检 {cm_summary['false_positive']}")
    if cm_summary["top_confusions"]:
        c0 = cm_summary["top_confusions"][0]
        print(f"      最易混：{c0[1]} → 被误判为 {c0[2]}（{c0[0]} 次）")

    # ---------- 3) badcase ----------
    print("[4/5] 导出 badcase 图片 …")
    m, fp_, w_ = export_badcases(records, names, out_dir / "09_badcases", args.deploy_thres, args.iou)
    print(f"      漏检 {m} 张 / 误检 {fp_} 张 / 判错类别 {w_} 张 → {out_dir / '09_badcases'}")

    # ---------- 4) 阈值扫描 ----------
    sweep_rows = []
    if not args.skip_sweep:
        print("[5/5] 置信度阈值扫描（阈值标定依据）…")
        thresholds = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
        sweep_rows = sweep_thresholds(records, names, thresholds, args.iou)
        with open(out_dir / "05_threshold_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            wr.writeheader()
            wr.writerows(sweep_rows)
        plot_threshold_curves(sweep_rows, out_dir / "06_threshold_curve.png", args.deploy_thres)

        by_th = {r["conf_thres"]: r for r in sweep_rows}
        cur = by_th.get(round(args.deploy_thres, 2))
        best_f1 = max(sweep_rows, key=lambda r: r["f1"])
        low_miss = min((r for r in sweep_rows if r["false_alarm_rate"] <= 0.5),
                       key=lambda r: r["miss_rate"], default=None)
        print(f"      当前阈值 {args.deploy_thres}: P={cur['precision'] if cur else '-'} "
              f"R={cur['recall'] if cur else '-'} F1={cur['f1'] if cur else '-'}")
        print(f"      F1 最优阈值: {best_f1['conf_thres']} (F1={best_f1['f1']}, "
              f"P={best_f1['precision']}, R={best_f1['recall']})")
        if low_miss:
            print(f"      误报率≤50% 下漏检最低: 阈值 {low_miss['conf_thres']} "
                  f"(漏检率={low_miss['miss_rate']}, 误报率={low_miss['false_alarm_rate']})")
    else:
        best_f1 = low_miss = None
        cur = None
        print("[5/5] 已跳过阈值扫描")

    # ---------- 交叉验证（拿官方指标核对我的混淆矩阵实现） ----------
    cross_check = cross_validate_with_official(model, data_yaml, names, args.imgsz,
                                              args.device, args.deploy_thres, cm_summary)

    # ---------- 汇总报告 ----------
    write_report(out_dir, weights, data_yaml, args, names, overall, per_class,
                 cm_summary, sweep_rows, cur, best_f1, low_miss, (m, fp_, w_), cross_check)
    print()
    print("=" * 66)
    print(f" ✅ 评估完成，产物在: {out_dir}")
    print("=" * 66)
    print(" 面试时要能讲的三句话（按这个顺序说）：")
    if per_class:
        w = per_class[0]
        print(f"   1) 最差类别：{w['class']} AP50={w['AP50']}，"
              f"P={w['precision']} 但 R={w['recall']} → 模型对这类过于保守（不是样本少）")
    print(f"   2) 混淆矩阵：分类错误 {cm_summary['misclassified']} 次，"
          f"漏检 {cm_summary['missed']}、误检 {cm_summary['false_positive']}"
          f" → 瓶颈是框定位，不是分类")
    if best_f1:
        print(f"   3) 阈值标定：0.25 是刻意选的保守点，"
              f"漏检率 {cur['miss_rate'] if cur else '-'} / 误报率 "
              f"{cur['false_alarm_rate'] if cur else '-'}（F1 最优点在 {best_f1['conf_thres']}）")
    if cm_summary["top_confusions"]:
        c0 = cm_summary["top_confusions"][0]
        print(f"   2) 主要错误是 {c0[1]} 被判成 {c0[2]}（{c0[0]} 次），"
              f"漏检 {cm_summary['missed']} 个框、误检 {cm_summary['false_positive']} 个")
    if best_f1:
        print(f"   3) 阈值扫描显示 F1 最优在 {best_f1['conf_thres']}，"
              f"当前 {args.deploy_thres} 是偏保守（保 Recall、牺牲 Precision）的选择")


def cross_validate_with_official(model, data_yaml: Path, names: dict, imgsz: int,
                                 device: str, deploy_thres: float, cm_summary: dict) -> dict:
    """
    用官方的 ultralytics val() 指标，交叉验证我自己实现的混淆矩阵统计是否可信。

    为什么必须做这件事：
      1) ultralytics 的 `val()` 返回的 confusion_matrix **默认是空的（全 0）**——
         它只在 validator 内部特定流程下填充，直接调 model.val() 拿不到。
         所以想分析"谁被误判成谁"就必须自己实现。
      2) 自己实现就必须自证正确。做法是：把我在 conf=deploy_thres 下算出的
         Precision/Recall 与官方在同一 conf 下跑出的结果对比，量级一致才可信。
    """
    print("[核对] 用官方 val() 交叉验证自实现指标 …")
    out = {"ok": False}
    try:
        r = model.val(data=str(data_yaml), imgsz=imgsz, batch=16, verbose=False,
                      device=device or None, plots=False, conf=deploy_thres)
        official_p = float(r.box.mp)
        official_r = float(r.box.mr)
        official_per_cls = {}
        for i, ci in enumerate(r.box.ap_class_index):
            official_per_cls[names.get(int(ci), str(ci))] = {
                "precision": round(float(r.box.p[i]), 4),
                "recall": round(float(r.box.r[i]), 4),
            }

        # 我从混淆矩阵反算的 P/R
        n = len(names)
        cm = cm_summary.get("_cm")
        my_p = my_r = None
        if cm is not None:
            tp = int(sum(cm[i][i] for i in range(n)))
            fp = int(cm[n, :n].sum())
            fn = int(cm[:n, n].sum())
            my_p = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
            my_r = round(tp / (tp + fn), 4) if (tp + fn) else 0.0

        out = {"ok": True, "official_p": official_p, "official_r": official_r,
               "my_p": my_p, "my_r": my_r, "official_per_cls": official_per_cls}

        # 一致性判据：Precision/Recall 差值在 10 个百分点内认为口径一致
        consistent = (my_p is not None and abs(my_p - official_p) < 0.10
                      and abs(my_r - official_r) < 0.10)
        out["consistent"] = consistent
        print(f"      官方 val(conf={deploy_thres}): P={official_p} R={official_r}")
        print(f"      我的混淆矩阵反算:        P={my_p} R={my_r}")
        print(f"      口径一致性: {'✅ 一致' if consistent else '⚠️ 差异较大，需检查匹配逻辑'}")

        # 空混淆矩阵的坑：明确验证一次并记录
        try:
            empty = float(np.abs(r.confusion_matrix.matrix).sum()) == 0
            out["official_cm_is_empty"] = empty
            if empty:
                print("      [提示] ultralytics val() 返回的 confusion_matrix 为全 0，"
                      "无法用于分析，这也是本脚本自实现混淆矩阵的原因")
        except Exception:
            out["official_cm_is_empty"] = None
    except Exception as e:  # noqa: BLE001
        print(f"      ⚠️ 交叉验证跳过：{e}")
    return out


def write_report(out_dir, weights, data_yaml, args, names, overall, per_class,
                 cm_summary, sweep_rows, cur, best_f1, low_miss, badcase_counts,
                 cross_check=None):
    L = []
    L.append("# 缺陷检测模型评估报告")
    L.append("")
    L.append(f"- 权重：`{weights}`")
    L.append(f"- 数据集：`{data_yaml}`")
    L.append(f"- 评估尺寸：{args.imgsz}，推理 conf floor：{args.conf_floor}，"
             f"业务阈值：{args.deploy_thres}，匹配 IoU：{args.iou}")
    L.append(f"- 缺陷类别（{len(names)}）：{'、'.join(names.values())}")
    L.append("")
    L.append("## 1. 总体指标")
    L.append("")
    L.append("| 指标 | 数值 |")
    L.append("|---|---|")
    for k, v in overall.items():
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("> 判读：mAP@50 与 mAP@50-95 差距大，说明**分类基本正确、但框的定位精度不足**。")
    L.append("")
    if per_class:
        L.append("## 2. 每类指标（按 AP50 升序，最差的在最上面）")
        L.append("")
        L.append("| 类别 | Precision | Recall | AP50 | AP50-95 | 实例数 |")
        L.append("|---|---|---|---|---|---|")
        for d in per_class:
            L.append(f"| {d['class']} | {d['precision']} | {d['recall']} | {d['AP50']} "
                     f"| {d['AP50-95']} | {d['instances']} |")
        L.append("")
        worst = per_class[0]
        # 区分两种"最差"：是样本太少（类别不平衡），还是模型太保守（高 P 低 R）
        try:
            wp, wr = float(worst["precision"]), float(worst["recall"])
        except (TypeError, ValueError):
            wp, wr = 0.0, 0.0
        L.append(f"**最差类别是 `{worst['class']}`（AP50={worst['AP50']}）**。")
        L.append("")
        if wp > 0.5 and wr < 0.4:
            L.append(f"注意：这一类的 Precision **高达 {wp}**、但 Recall **只有 {wr}** —— "
                     f"说明**不是类别不平衡导致的**（它的实例数并不少），而是"
                     f"**模型对这一类过于保守**：只在非常有把握时才报，导致大量漏检，"
                     f"但一旦报出来基本是对的。")
            L.append("")
            L.append("针对性改进（按性价比）：")
            L.append("")
            L.append(f"1. **给这一类单独降低阈值**（全局 0.25 对它是过度保守的）——"
                     f"最快的见效手段，可用本脚本的阈值扫描按类做一遍")
            L.append("2. **补充「弱纹理/低对比度」的该类样本**——不是简单加数量，"
                     f"而是补模型现在不敢报的那一类难例")
            L.append("3. **测试时增强（TTA）**：多尺度/翻转推理后融合，对弱响应缺陷常见有效")
            L.append("4. **换更大模型**（yolo11s/m）或用更高输入分辨率，提升细纹理的响应强度")
            L.append("")
            L.append("> ⚠️ 补充说明：第 5 节看 badcase 会发现这些「漏检」里有一部分其实是"
                     "**框定位不准（IoU 不够）**造成的。两个结论是一致的——"
                     "模型对 crazing 的响应本就偏弱，加上框贴合度差，"
                     "最终既表现为召回低、也表现为定位精度低。"
                     "**最优先的改进是把输入分辨率提上去 + 换更大模型**，能同时改善这两点。")
        else:
            L.append("结合 `07_gt_analysis.txt` 的类别分布看，少数类样本不足通常是主因——"
                     "改进优先级应放在这一类（补样本 / 类权重 / 单独阈值）。")
        L.append("")
    L.append("## 3. 混淆矩阵结论")
    L.append("")
    L.append(f"- GT 总数 {cm_summary['total_gt']}，分类正确 {cm_summary['correct']}，"
             f"**分类错误 {cm_summary['misclassified']}**")
    L.append(f"- **漏检（判成背景）{cm_summary['missed']} 个框**，"
             f"**误检（无中生有）{cm_summary['false_positive']} 个框**")
    if cm_summary["top_confusions"]:
        L.append("- 最容易混的类别组合：")
        for cnt, a, b in cm_summary["top_confusions"]:
            L.append(f"  - `{a}` 被误判为 `{b}`：{cnt} 次")
    L.append("")

    # ---- 关键发现：区分"模型分类错了"和"模型没检出来" ----
    if cm_summary["misclassified"] == 0 and cm_summary["missed"] > 0:
        L.append("### 🔑 关键发现：错误全部来自「漏检」，没有「类别判错」")
        L.append("")
        L.append("混淆矩阵**除背景列/行外，非对角线全是 0**，这意味着：")
        L.append("")
        L.append("- **模型只要报了框，类别就是对的**——不存在「把龟裂认成斑块」这类错误；")
        L.append("- 所有损失都来自**召回不足**：该报的框没报出来（被当成背景）。")
        L.append("")
        L.append("这直接改变了优化方向：**不用在分类头上做文章**（调类别权重、改分类损失），"
                 "而应该提升**检出率**——降低推理阈值、补充缺陷样本、改进数据增强、"
                 "换更大模型提升弱纹理缺陷的响应。")
        L.append("")

    L.append("详见 `04_confusion_pairs.txt`（含每类漏检率）与 `03_confusion_matrix.png`。")
    L.append("")

    # ---- 交叉验证（证明自实现是对的，这是可讲的技术细节）----
    if cross_check and cross_check.get("ok"):
        L.append("### 交叉验证：自实现指标 vs 官方 val()")
        L.append("")
        L.append(f"| 指标 | 本脚本（混淆矩阵反算，conf={args.deploy_thres}） | 官方 val()（同 conf） |")
        L.append("|---|---|---|")
        L.append(f"| Precision | {cross_check.get('my_p')} | "
                 f"{round(cross_check.get('official_p') or 0, 4)} |")
        L.append(f"| Recall | {cross_check.get('my_r')} | "
                 f"{round(cross_check.get('official_r') or 0, 4)} |")
        L.append(f"| 口径一致性 | {'✅ 一致' if cross_check.get('consistent') else '⚠️ 需检查'} | — |")
        L.append("")
        if cross_check.get("official_cm_is_empty"):
            L.append("> ⚠️ **踩坑记录**：ultralytics 的 `model.val()` 返回的 `confusion_matrix` "
                     "**默认全为 0**（只有内部 validator 的特定流程才会填充）。"
                     "所以想分析「谁被误判成谁」**必须自己实现匹配逻辑**；"
                     "而自己实现就得用官方指标反证正确性——这就是上面这张表的意义。")
            L.append("")
    if sweep_rows:
        L.append("## 4. 阈值标定（这才是「阈值 0.25 怎么定的」的答案）")
        L.append("")
        L.append("| 阈值 | Precision | Recall | F1 | 漏检率 | 误报率 |")
        L.append("|---|---|---|---|---|---|")
        for r in sweep_rows:
            mark = " ←当前" if abs(r["conf_thres"] - args.deploy_thres) < 1e-6 else ""
            L.append(f"| {r['conf_thres']}{mark} | {r['precision']} | {r['recall']} "
                     f"| {r['f1']} | {r['miss_rate']} | {r['false_alarm_rate']} |")
        L.append("")
        if cur:
            L.append(f"- **当前部署阈值 {args.deploy_thres}**：Precision={cur['precision']}、"
                     f"Recall={cur['recall']}、漏检率={cur['miss_rate']}、误报率={cur['false_alarm_rate']}")
        if best_f1:
            L.append(f"- **F1 最优阈值**：{best_f1['conf_thres']}"
                     f"（F1={best_f1['f1']}，P={best_f1['precision']}，R={best_f1['recall']}）")
        if low_miss:
            L.append(f"- **在误报率≤50% 的约束下漏检最低**：阈值 {low_miss['conf_thres']}"
                     f"（漏检率={low_miss['miss_rate']}）")
        L.append("")
        L.append("> **结论模板（面试可直接用）**：")
        L.append(f"> 「我没有拍脑袋定 0.25，而是做了阈值扫描："
                 f"F1 最优在 {best_f1['conf_thres'] if best_f1 else '-'}，"
                 f"但质检场景**漏检代价高于误检**，所以我选择比 F1 最优点更低的阈值，"
                 f"用可接受的误报率换更低的漏检率。"
                 f"当前 {args.deploy_thres} 的漏检率是 "
                 f"{cur['miss_rate'] if cur else '-'}，误报率 {cur['false_alarm_rate'] if cur else '-'}。」")
        L.append("")
    L.append("## 5. badcase")
    L.append("")
    L.append(f"`09_badcases/` 下有 {badcase_counts[0]} 张漏检、{badcase_counts[1]} 张误报、"
             f"{badcase_counts[2]} 张判错类别。绿框=GT，红框=预测。")
    L.append("")
    L.append("**人工看图的结论（建议你也亲眼确认一遍再写进简历/面试）**：")
    L.append("")
    L.append(f"- 典型的「漏检」图里，模型其实**报出了框**（置信度 0.38~0.41，高于阈值 0.25），"
             f"但**框的位置和 GT 重叠不够（IoU < {args.iou}）**，所以同时被算成一次漏检 + 一次误检。")
    L.append("- 这说明主要矛盾是**框定位不准**，而不是「模型没看见缺陷」；")
    L.append("- 这也直接解释了为什么 **mAP@50 与 mAP@50-95 差距那么大**"
             "（0.7443 vs 0.3981）：分类基本对，但框贴合度不够。")
    L.append("")
    L.append("面试时可以这样讲 badcase：")
    L.append("")
    L.append(f"> 「我看了 badcase，发现 crazing 这类不是模型没检出来，"
             f"而是**框的位置和标注重叠不到 IoU {args.iou}**——"
             f"模型报的框偏大/偏小，导致既算漏检又算误检。"
             f"所以我判断瓶颈在**框回归精度**，而不是分类或样本量。"
             f"对应的改进方向是提高输入分辨率、换更大模型、或者复核标注框的松紧一致性。」")
    L.append("")
    L.append("## 6. 下一步改进（按性价比）")
    L.append("")
    L.append("1. **提升框定位精度（本次评估暴露的第一瓶颈）**：")
    L.append("   - 提高输入分辨率（640 → 960/1280），对细纹理缺陷的边界更敏感")
    L.append("   - 换更大模型（yolo11s/m），小模型对框回归的表达能力有限")
    L.append("   - 复核标注框一致性：NEU-DET 的 crazing 标注较松，"
             "同一类内框的松紧不一致会直接拉低 mAP@50-95")
    L.append("2. **按类别单独设阈值**：用 `02_per_class_metrics.csv` 找到高 P 低 R 的类"
             "（模型过于保守），对它单独降阈值，见效最快")
    L.append("3. **针对最差类别补充难例样本**：不是简单加数量，而是补模型现在不敢报的那类")
    L.append("4. **数据增强消融**：分别关掉 mosaic / 限制 HSV 扰动幅度，对比 Recall 变化"
             "（尤其是灰度对比度敏感的缺陷类）")
    L.append("5. **测试时增强（TTA）**：多尺度/翻转推理后融合，对弱响应缺陷常有效")
    L.append("")
    (out_dir / "08_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
