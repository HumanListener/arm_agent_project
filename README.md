# 基于多模态智能体的工业缺陷检测与分拣系统

> 构建一个 AI 智能体：它"看到"工业零件（钢板）表面的缺陷 → 理解缺陷类型 → 自主决策调用工具
> （MCP 机械臂）把缺陷件分拣出去。完整复刻「**感知 — 决策 — 行动**」的具身智能闭环。

```
                    ┌──────────────────────────── 感知层 ────────────────────────────┐
 图片/摄像头/虚拟相机 ──▶  YOLOv8 (NEU-DET 六类缺陷)  ──▶  OpenCV ROI/HSV 预处理 + 像素→世界坐标标定
                    └───────────────────────────────────────────────────────────────┘
                                              │  结构化检测结果 {类别, 置信度, bbox, 世界坐标}
                                              ▼
                    ┌──────────────────────────── 决策层 ────────────────────────────┐
                    │  LangGraph 状态图：perceive → decide → act → finish            │
                    │  决策引擎 = 规则知识库（永远可用） + 本地 Ollama 大模型（可选增强）│
                    └───────────────────────────────────────────────────────────────┘
                                              │  {动作 SORT/REVIEW/PASS, 投放区, 理由, 决策报告}
                                              ▼
                    ┌──────────────────────────── 执行层 ────────────────────────────┐
                    │  MCP Server（mcp_arm_tools.py）：move_arm_to / sort_defect ... │
                    │        │ stdio (MCP 协议)                                      │
                    │        ▼                                                       │
                    │  FastAPI 指令总线 :8000  ◀──轮询──  浏览器虚拟机械臂(Three.js) │
                    │                                  二连杆 IK 真解算 + 抓取/投放  │
                    └───────────────────────────────────────────────────────────────┘
                                              ▲
                    ┌──────────────────────────── 交互层 ────────────────────────────┐
                    │  Streamlit：上传图片 / 摄像头拍照 → 检测 → 决策 → 跳转机械臂页 │
                    └───────────────────────────────────────────────────────────────┘
```

---

## 1. 快速开始

### 1.0 以后每天怎么启动（推荐看这一节）

**方式 A：一条命令（最省事，推荐）**

```bash
cd /Users/sq/PycharmProjects/arm_agent_project
./start.sh                 # 自动找环境 → 起 FastAPI + Streamlit → 自动打开两个浏览器页面
```

| 命令 | 作用 |
|---|---|
| `./start.sh` | 启动全部服务并自动打开浏览器 |
| `./start.sh --no-browser` | 只启动服务，不开浏览器 |
| `./start.sh --status` | 查看 FastAPI / Streamlit / Ollama 是否在跑 |
| `./start.sh --stop` | 停止本项目启动的服务（释放 8000 / 8501 端口） |

`start.sh` 会自动探测带 `streamlit/fastapi/opencv` 的 Python 环境（优先 `conda envs/langgraph`），
已经起来的服务不会重复启动，**重复执行是安全的**。日志写在 `dataset_temp/api_server.log`
和 `dataset_temp/streamlit.log`。

**方式 B：PyCharm 里点运行按钮**

`.idea/runConfigurations/` 里已配好 4 个运行配置，打开 PyCharm 右上角下拉框即可选：

| 配置名 | 作用 |
|---|---|
| `1. 一键启动全部服务` | 等同于 `./start.sh`（推荐） |
| `2. Streamlit 检测主界面` | 只起 Streamlit 界面 |
| `3. FastAPI 总线 + 机械臂页面` | 只起 FastAPI（带 `--reload`，改后端代码自动重载） |
| `4. 全链路自检` | 跑 `tools/verify_all.py`，13 项检查 |

> ⚠️ **重要**：本项目用的是 conda 环境 `langgraph`，而 PyCharm 工程默认解释器是 `/opt/anaconda3`（base，**没装** fastapi/opencv/langgraph）。
> 所以：
> - 用上面 4 个运行配置 → 已在配置里写死用 `langgraph`，直接点就行；
> - 如果你想在编辑器里直接右键运行 `ui_streamlit/streamlit_app.py` 之类的文件 → 先改解释器：
>   `PyCharm → Settings → Project: arm_agent_project → Python Interpreter → Add Interpreter → Conda → 选 langgraph`。

**方式 C：手动开两个终端**

```bash
conda activate langgraph

# 终端 1：FastAPI 指令总线 + 浏览器虚拟机械臂页面
uvicorn api_server.arm_fastapi_server:app --port 8000 --reload
#   机械臂页面  http://127.0.0.1:8000/sim/
#   接口文档    http://127.0.0.1:8000/docs

# 终端 2：Streamlit 交互主界面
streamlit run ui_streamlit/streamlit_app.py
#   http://localhost:8501
```

**打开哪两个页面**

| 页面 | 地址 | 用途 |
|---|---|---|
| ① 检测主界面 | <http://localhost:8501> | 上传图片 / 摄像头 / 示例图 → 检测 → 看决策报告 |
| ② 机械臂页面 | <http://127.0.0.1:8000/sim/> | 虚拟机械臂，检测到缺陷后从这里自动分拣 |

> 本地大模型想启用就再开一个终端：`OLLAMA_ORIGINS='*' ollama serve`。
> 不启动也能跑，只是决策降级为规则知识库（关掉界面侧边栏的「调用本地大模型决策」开关同理）。

### 1.1 环境（本项目实测可用）

```bash
conda activate langgraph          # 依赖已装：langgraph / ultralytics / streamlit / fastapi / mcp / opencv
# 若换新环境：
pip install -r requirements.txt
```

### 1.2 前置：本地大模型

```bash
OLLAMA_ORIGINS='*' ollama serve    # 允许浏览器跨域访问
ollama list                        # 本机已有：deepseek-r1:8b / deepseek-v4-flash:cloud / deepseek-v4-pro:cloud
```

> - `config/settings.py` 里 `OLLAMA_MODEL` 默认 `deepseek-r1:8b`，会自动探测本机实际存在的模型并回退。
> - **CPU 上 8B 模型单次决策约 60~90 秒**。所以决策层默认走"规则知识库 +（可选）大模型"：
>   界面侧边栏有「调用本地大模型决策」开关，关掉时秒级出结果，打开时给出大模型版的判据。
> - Ollama 没启动也能全程演示：自动降级为规则决策，只影响"决策话术"，不影响闭环。

### 1.3 另一种启动方式（Python 脚本入口）

不想用 shell 脚本时，`main.py` 也能启动（内部会自动拉起 FastAPI）：

```bash
python main.py --ui          # 等价于 ./start.sh，只是不自动开浏览器
python main.py --api         # 只起 FastAPI
python main.py --demo        # 跑一轮网格巡检
python main.py               # 打印启动指引
```

### 1.4 演示前自检（一条命令确认四层都通）

```bash
python tools/verify_all.py              # 配置/服务/感知/决策/MCP执行/LangGraph/日志 共 13 项
python tools/verify_all.py --with-llm   # 额外验证本地大模型决策（慢，约 1~2 分钟）
node  tools/ik_selftest.mjs             # 机械臂 IK 精度与关节限位
```

---

## 2. 演示流程：图片检测 → 自动跳转机械臂分拣

这是本项目的主线，全部在 Streamlit 页面完成：

1. **选择图像来源**（三种，无需摄像头也能跑）
   - `📁 上传图片`：拖入任意钢板表面图片；
   - `🎥 摄像头拍照`：浏览器调摄像头（localhost 下可用，不装本机摄像头驱动也行）；
   - `🗂️ 示例图片`：内置 `samples/`，`demo_defect_high.jpg` 是高置信缺陷样本，**一定会触发分拣**。
2. **点击「🚀 开始检测」**，Agent 依次执行：
   - `perceive`：YOLO 推理 + 画框可视化 + 像素坐标 → 工作台世界坐标；
   - `decide`：规则知识库给底线结论，若开启大模型再让 Ollama 组织判据，输出中文决策报告
     （例如："检测结论：夹杂（inclusion），置信度 91.1%，严重度 高 / 处置：执行【分拣】→ 投放至「废品区」"）；
   - `act`：**通过 MCP 协议**调用机械臂工具 `sort_defect(x, y, dest)`。
3. **界面出现缺陷告警 + 决策报告 + MCP 工具调用轨迹**（每次调用的参数、耗时、返回值都可见）。
4. **跳转浏览器机械臂页面**：页面提供大按钮 `🚀 打开浏览器机械臂页面（自动执行分拣）`，
   也可以在页面内勾选「📺 在页面内同屏显示机械臂」直接 iframe 同屏演示。
   页面 URL 形如 `/sim/?embed=1&auto=1`。
5. 机械臂页面（Three.js）轮询总线 → 拿到 `pick_task` → **用二连杆 IK 反解出 6 个关节角** →
   执行"抬起 → 悬停到缺陷坐标 → 下降 → 夹紧 → 抬起 → 平移到目标区 → 放下 → 回零"，并回调
   `/api/arm_ready` 通知 Agent 动作完成。

> **要点**：分拣任务挂在 FastAPI 总线上（`pick_task`），即使浏览器页面还没打开也不会丢任务——
> 页面一打开就自检并补做，所以"Streamlit 检测完 → 再打开机械臂页面"的顺序完全没问题。

三个目标区（`config/settings.py` 的 `DEST_BINS`，与页面圆盘严格一致）：

| 区域 | 坐标 (x, y) 米 | 触发条件 |
|---|---|---|
| 废品区 ng | `(0.20, 0.00)` | 不可返修的严重缺陷：夹杂、压入氧化铁皮、高置信划痕 |
| 返修区 rework | `(0.00, 0.05)` | 可打磨返修：龟裂、斑块、浅麻点 |
| 复检区 review | `(-0.20, 0.00)` | 置信度不足或严重度拿不准 → 人工复检，不冒险判废 |

---

## 3. 其它两种玩法

### 3.1 自动网格巡检（虚拟相机构图）

打开机械臂页面后，在 Streamlit「📊 巡检看板」点「▶️ 启动一轮完整巡检」：

```
capture(浏览器拍照) → perceive(YOLO) → decide → PICK/REVIEW 走 act，否则 next_grid → … → done(回零)
```

`agent/agent_main.py` 里的 LangGraph 会在 `INSPECT_GRID`（工作台上 3×3 点位）上循环，
每个点拍一张、检一次、判一次，日志写入 SQLite（`db/inspection.db`），看板实时展示。

命令行等价写法：`python main.py --demo`

### 3.2 MCP 工具面板 / 命令行

在 Streamlit「🧰 MCP 工具面板」可以现场 `list_tools` 并调用任意工具（比如 `go_home`、`sort_defect`），
直观展示"大模型通过 MCP 协议驱动机械臂"这一层。

命令行：

```bash
python -m mcp_service.mcp_client                              # 列出 MCP 工具
python -m mcp_service.mcp_client sort_defect '{"x":0.01,"y":0.0,"dest":"ng"}'   # 调一个工具
python -m mcp_service.mcp_arm_tools --http                    # 以 streamable-http 方式跑 MCP Server
python main.py --image samples/demo_defect_high.jpg           # 单图闭环（不经过浏览器界面）
```

---

## 4. 目录结构

```
arm_agent_project/
├── start.sh                         # 【一键启动】./start.sh / --stop / --status
├── main.py                          # 统一入口：--ui / --api / --demo / --image
├── config/settings.py               # 全局配置：Ollama、端口、阈值、网格、分拣区、MCP 路径
│
├── yolo_module/
│   ├── yolo_defect_infer.py         # 感知层：YOLOv8 推理 + OpenCV 预处理 + 像素→世界坐标
│   └── weights/best.pt              # NEU-DET 六类缺陷权重（crazing/inclusion/patches/pitted/scale/scratches）
│
├── agent/
│   ├── decision.py                  # 决策层：规则知识库 + Ollama 决策 + 中文报告模板 + 自动降级
│   ├── inspect_agent.py             # LangGraph 单图闭环（Streamlit 检测走这里，act 走 MCP）
│   └── agent_main.py                # LangGraph 多点网格巡检（capture/perceive/decide/act/next_grid）
│
├── mcp_service/
│   ├── mcp_arm_tools.py             # 【执行层】MCP Server：arm 工具集（本项目的"手"）
│   ├── mcp_client.py                # 标准 MCP 客户端（stdio），供 Agent / 界面调用工具
│   └── mcp_logger_server.py         # 巡检日志 MCP Server（SQLite 持久化）
│
├── api_server/
│   ├── arm_fastapi_server.py        # 指令总线：/api/bus /api/pick /api/upload_frame /api/detect ...
│   └── arm_client.py                # 总线的 Python 客户端（MCP Server 内部用它真正下发动作）
│
├── sim_web_static/index.html        # 【浏览器虚拟机械臂】Three.js 6 轴 + 二连杆 IK + 抓取投放
├── ui_streamlit/
│   ├── streamlit_app.py             # 【交互主界面】上传/摄像头/示例图 → 检测 → 决策 → 跳转分拣
│   ├── app.py                       # 巡检看板（网格巡检、日志、总线状态）
│   └── common.py                    # 服务自检/自动拉起、Agent 调用、总线访问
│
├── samples/                         # 内置演示图片（tools/make_samples.py 可重新生成）
├── tools/
│   ├── verify_all.py                # 全链路自检：配置/服务/感知/决策/MCP/LangGraph/日志
│   ├── ik_selftest.mjs              # 从 index.html 抽取真实 IK 代码做自检（Node，零依赖）
│   ├── ik_workspace.mjs             # 机械臂可达工作空间地图（改分拣区坐标前先跑它）
│   └── make_samples.py              # 生成演示样例图并实测检测结果
├── .idea/runConfigurations/         # PyCharm 一键运行配置（4 个入口）
├── db/inspection.db                 # SQLite 巡检日志
└── dataset_temp/                    # 运行产物：上传帧、可视化图、FastAPI 日志
```

---

## 5. 关键技术点（答辩/写报告可直接引用）

| 模块 | 做了什么 | 代码位置 |
|---|---|---|
| **感知** | NEU-DET 六类缺陷检测；模型侧用很低的置信度 floor（0.05）召回候选，业务层再用可调阈值判定，避免"中间态"被模型直接丢掉 | `yolo_module/yolo_defect_infer.py` |
| **虚拟标定** | 检测框中心像素 → 工作台物理坐标线性映射（真实产线换相机内外参标定即可） | 同上 `pixel_to_world()` |
| **决策** | 两级决策：规则知识库（确定性、永远可用）+ Ollama（更拟人的判据），大模型输出非法动作/坑位时自动校正回规则结论 | `agent/decision.py` |
| **状态机** | LangGraph `StateGraph`，条件边按决策分流到分拣/复检/放行；单图与网格巡检两条图复用同一决策引擎 | `agent/inspect_agent.py`、`agent/agent_main.py` |
| **工具调用** | 真正的 MCP 协议（stdio，`mcp` 1.x）：Agent 侧不是 import 函数，而是 `list_tools` / `call_tool`，轨迹入库可回放 | `mcp_service/mcp_arm_tools.py`、`mcp_client.py` |
| **执行层桥接** | MCP 工具 → HTTP → 指令总线（进程内单例 + 线程锁）→ 浏览器轮询；任务持久挂在总线上，页面晚打开也不丢 | `api_server/arm_fastapi_server.py` |
| **机械臂运动学** | 二连杆解析 IK + 腕部偏移迭代补偿 + 双肘分支择优，实测末端误差 < 2mm；关节限位硬约束 | `sim_web_static/index.html` `ikArm()` |
| **交互** | Streamlit 三来源输入（上传/摄像头/示例）、决策报告、MCP 调用轨迹、跨页跳转 + iframe 同屏；FastAPI 掉线自动拉起 | `ui_streamlit/streamlit_app.py`、`common.py` |

### 为什么不用 `use_container_width`、为什么大模型只输出短 JSON？

- Streamlit 1.64 起 `use_container_width` 已弃用，统一改用 `width='stretch'`。
- 本地 8B 模型生成整段中文报告要 1~2 分钟；因此只让模型输出 `{action, dest, reason, severity}`
  这几十个 token，**完整中文决策报告由 Python 模板拼装**，速度与可读性兼得（见 `_compose_report()`）。

### 关于机械臂可达性（改坐标前必读）

虚拟机械臂底座在世界坐标 `(0, 0.06, -0.25)`，大臂 0.15m + 小臂 0.15m + 夹爪 0.06m，
关节限位 ±1.57 rad，因此在工作台上的可达区域是以底座为圆心、半径约 0.10~0.34m 的扇形，
**底座正后方（z < -0.20）基本够不到**。三个分拣区坐标就是按这个约束挑的：

```bash
node tools/ik_workspace.mjs     # 打印可达性地图，改 DEST_BINS 前先跑一遍
node tools/ik_selftest.mjs      # 直接抽取 index.html 里真实运行的 IK 做自检
```

---

## 6. API 速查（FastAPI 指令总线）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（Streamlit 用它决定是否自动拉起服务） |
| GET | `/api/bus` | 浏览器轮询：拍照请求 / 目标点 / 分拣任务 / 决策广播 |
| POST | `/api/dispatch` | 下发末端目标点 `{x, y, z}` |
| POST | `/api/pick` | 下发完整分拣任务 `{x, y, dest_x, dest_y, note}` |
| POST | `/api/go_home` | 回零 |
| POST | `/api/request_capture`、`/api/upload_frame` | 请求拍照 / 浏览器上传虚拟相机帧 |
| POST | `/api/arm_ready` | 浏览器上报"动作完成" |
| POST | `/api/decision` | 决策结果广播（浏览器 HUD 同步显示） |
| POST | `/api/detect` | 浏览器页面自带的图片上传检测（返回画框 base64 + 规则决策） |
| GET | `/sim/` | 浏览器虚拟机械臂页面 |

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 页面提示 FastAPI 不可用 | Streamlit 会尝试自动拉起；失败时手动 `uvicorn api_server.arm_fastapi_server:app --port 8000`，日志在 `dataset_temp/api_server.log` |
| 8000 端口被占用 | `lsof -ti tcp:8000 \| xargs kill -9`，或在 `config/settings.py` 改端口 |
| 检测到缺陷但机械臂不动 | 机械臂页面没打开（Agent 侧会打印"N 秒内没等到 arm_ready"）。打开 `/sim/?auto=1` 后任务会自动补做 |
| 决策耗时 60~90 秒 | 本地 8B 模型 CPU 推理，属正常；关掉侧边栏「调用本地大模型决策」即秒出规则结论 |
| Ollama 连不上 / 模型不存在 | `OLLAMA_ORIGINS='*' ollama serve`；`config/settings.py` 的 `OLLAMA_MODEL` 会自动回退到本机已有模型 |
| 点数/坐标对不上 | 先跑 `node tools/ik_workspace.mjs` 确认目标点在可达区域内 |
| 想换自己的 YOLO 权重 | 覆盖 `yolo_module/weights/best.pt`；类别名从模型里读，新类别会走"未知类别 → 人工复检"兜底 |
| 没有外网导致页面空白 | Three.js 走 CDN，把 `three.module.js` 下载到 `sim_web_static/` 并改 importmap 路径 |
