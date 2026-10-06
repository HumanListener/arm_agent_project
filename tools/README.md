# 工具脚本（开发/验证用，不参与运行时）

## 1. `ik_selftest.mjs` —— 机械臂 IK 自检（**改坐标后必跑**）

直接从 `sim_web_static/index.html` 里**抽取真实运行的 IK 代码**（`ikArm` / `poseToReach`），
再用独立实现的正运动学回算末端位置做交叉验证。所以它验证的就是页面上真正跑的数学，
不存在"工具脚本和页面代码抄了两份、结论对不上"的问题。

```bash
node tools/ik_selftest.mjs
```

检查三项：
1. `DEST_BINS` 三个分拣坑位是否可达（误差 < 10mm）；
2. 关节限位是否守住 ±1.57（夹爪 [0, 1.57]）；
3. 钢板上的典型抓取点是否可达。

退出码 0 = 全部通过。

## 2. `ik_workspace.mjs` —— 可达工作空间地图

```bash
node tools/ik_workspace.mjs
```

打印工作台网格的可达性地图（`O` 可达 / `.` 不可达）与候选点的 IK 误差。
**修改 `config/settings.py` 的 `DEST_BINS` 之前先跑它**，避免选到"看起来合理但机械臂够不到"的点。

几何前提：底座世界坐标 `(0, 0.06, -0.25)`，大臂 0.15m、小臂 0.15m、夹爪 0.06m，关节限位 ±1.57 rad。
实测可达区域 ≈ 以底座为圆心、半径 0.10~0.34m 的扇形；底座正后方（z < -0.20）基本够不到。

## 3. `make_samples.py` —— 生成演示样例图

```bash
python tools/make_samples.py          # 生成 samples/ 里的三张样例并实测检测结果
python tools/make_samples.py --check  # 只跑模型确认现有样例的检测结果
```

| 文件 | 预期检测结果 | 演示用途 |
|---|---|---|
| `samples/demo_defect_high.jpg` | `inclusion` ≈ 0.91 → SORT 废品区 | 演示完整"检测 → 决策 → MCP 分拣" |
| `samples/demo_defect.jpg` | `rolled-in_scale` ≈ 0.19（低于阈值） | 演示"置信度不足 → 转人工复检" |
| `samples/demo_clean.jpg` | 无缺陷 | 演示"放行，不动机械臂" |

## 4. `evaluate_model.py` —— 模型评估 + 阈值标定（**补上项目最大的短板**）

```bash
# 完整评估（约 3~4 分钟）
/opt/anaconda3/envs/langgraph/bin/python tools/evaluate_model.py

# 指定模型/数据集
/opt/anaconda3/envs/langgraph/bin/python tools/evaluate_model.py \
    --weights yolo_module/weights/best.pt --data ~/Desktop/NEU-DET/data.yaml

# 快速验证（只跑 30 张，跳过阈值扫描）
/opt/anaconda3/envs/langgraph/bin/python tools/evaluate_model.py --limit 30 --skip-sweep
```

产出到 `evaluation/`：
- 总体与每类 Precision / Recall / AP50 / AP50-95
- **混淆矩阵**（含背景行/列，区分漏检与误检）+ 热力图
- **置信度阈值扫描**（0.05~0.95 的 P/R/F1/漏检率/误报率）+ 曲线图 ← 阈值标定依据
- badcase 图片（漏检 / 误报 / 判错类别，绿框=GT、红框=预测）
- 汇总报告 `08_report.md`

实测结论见 `evaluation/README.md`。**注意**：ultralytics `val()` 返回的
`confusion_matrix` 默认全为 0，所以本脚本自己实现了框匹配逻辑，
并用官方指标在同一置信度下做了交叉验证。
