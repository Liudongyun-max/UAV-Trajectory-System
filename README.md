# UAV Trajectory System (无人机轨迹预测与目标检测系统)

[![Git Flow: development](https://img.shields.io/badge/git--flow-development-orange.svg?style=flat-square)](./GIT_WORKFLOW.md)
[![Python: 3.10](https://img.shields.io/badge/python-3.10-blue.svg?style=flat-square)](https://www.python.org/)
[![NPU Acceleration: RK3588](https://img.shields.io/badge/NPU-RK3588_RKNN-green.svg?style=flat-square)](./yolov5_model/uav_yolo_rknn_report.md)
[![Code Lines: 15,798](https://img.shields.io/badge/code-15,798_lines-purple.svg?style=flat-square)](./PROJECT_DOCUMENTATION.md)
[![Episodes: 444](https://img.shields.io/badge/data-444_episodes-orange.svg?style=flat-square)](./export/)

**UAV Trajectory System** 是一套面向无人机 (UAV) 目标检测、时序轨迹跟踪与航迹预测的全栈系统。本项目涵盖了从仿真数据采集、数据质量审计、时序模型训练到边缘端实时部署的闭环工作流程。

---

## 核心特性 (Key Features)

* **多路 NPU 加速推理**：支持多路 USB 摄像头并发采集，基于 RK3588 NPU (RKNN) 进行底层硬件级加速推理。
* **级联检测管线**：融合"运动引导级联检测"（帧差分法 + LCM 对比度滤波）与 YOLOv5 深度精细识别，大幅提升远距离极小目标的捕获概率。
* **多特征轨迹跟踪 (Feature Tracker)**：
  * **轨迹去重融合**：支持 20px 检测质心融合与 35px 轨迹消重，自动剪除单目标分裂/多叉轨迹。
  * **一阶指数平滑滤波**：引入一阶低通滤波 (α = 0.65) 对 BBox 的中心点及宽高进行防抖修正，画面定位框锁定极其平稳。
* **GRU 时序预测网络**：
  * **轨迹真伪判决**：通过 GRU 神经网络对 20 帧历史滑动窗口轨迹进行真伪无人机目标分类（过滤强干扰弱噪点）。
  * **未来轨迹预测**：根据历史轨迹特征，预测无人机未来 5 帧的飞行航迹，显示黄色预测线。
* **距离估算系统**：物理基线 + MLP残差修正 + GRU时序精化三级距离估算，支持20-500m范围。
* **自动化数据审计**：包含独立的 8 阶段仿真 Episode 数据质量审计（A/B/C/D 四级评定），严格把控训练样本质量。

---

## 系统架构 (System Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          UAV Trajectory System                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │  数据采集层        │    │  质量分析层        │    │  模型训练层        │          │
│  │  Gazebo + PX4     │───>│  analyze_episode  │───>│  train/          │          │
│  │  5路摄像头仿真      │    │  batch_analyze    │    │  GRU Training    │          │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘          │
│           │                       │                       │                     │
│           v                       v                       v                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │  export/input/    │    │  export/outputs/  │    │  models/         │          │
│  │  444 Episodes     │    │  Analysis Reports │    │  *.pth *.pt      │          │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘          │
│                                                             │                   │
│                                                             v                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         推理与部署层                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │   │
│  │  │ gru detect/  │  │ yolov5+GRU/  │  │yolov5_model/ │  │distance  │  │   │
│  │  │ GRU独立推理   │  │ YOLO+GRU级联  │  │ RK3588边缘端  │  │model/    │  │   │
│  │  │ PyQt5 GUI    │  │ PyQt5 GUI    │  │ 5路实时检测    │  │距离估算   │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 项目目录结构

```text
.
├── .github/                   # GitHub 规范（PR 模板等）
├── agent/                     # 辅助自动化 Agent 脚本与技能定义
├── distance/                  # 距离标定与缩放比对照规范
├── distance model/            # 距离估算模块（物理基线+ML+GRU，含GUI与训练框架）
├── document/                  # 核心模型及时序轨迹网络设计说明文档
├── export/                    # [已忽略] 仿真输入数据集与批量分析输出报告
├── gru detect/                # GRU 独立轨迹预测模块 (含 PyQt5 GUI 与防抖追踪)
├── test  vedio/               # [已忽略] 测试视频数据集
├── train/                     # GRU 时序模型训练框架（包含数据集划分、多指标训练及模型转换）
├── yolov5+GRU/                # YOLOv5 + GRU 时序级联检测推理集成模块 (PyQt5 GUI)
├── yolov5_model/              # RK3588 边缘端多摄像头 NPU 部署方案与 UDP 通信模块
│
├── GIT_WORKFLOW.md            # 项目版本管理与 PR 提交流程规范 (开发必读)
├── PROJECT_DOCUMENTATION.md   # 项目系统架构全景审计说明书 (v7.0)
├── ROOT_FILES_ANALYSIS.md     # 根目录文件深度分析报告
└── run_yolo_gru_test.py       # YOLOv5 + GRU 级联推理演示启动脚本
```

---

## 快速上手 (Quick Start)

### 1. 环境准备
项目主要依赖 Python 3.10 环境：
```bash
# 创建并激活 conda 虚拟环境
conda create -n uav_gru python=3.10
conda activate uav_gru

# 安装核心依赖
pip install -r "gru detect/requirements.txt"
```

### 2. 启动 GRU 独立推理与防抖追踪窗口
```bash
cd "gru detect"
python predict_gui.py
# 在 GUI 界面选择测试视频与模型权重 (.pth/.onnx)，点击开始即可实时预览防抖追踪轨迹
```

### 3. 运行 YOLOv5 + GRU 级联推理
```bash
cd "yolov5+GRU"
python predict_gui.py
# 该模块将加载 YOLOv5 目标检测器与 GRU 预测网络，进行两阶段端到端无人机实景预测
```

### 4. 距离估算模块
```bash
cd "distance model"
python distance_model_gui.py
# 启动距离模型可视化界面，支持双模式检测（分割模式/YOLO模式）
```

### 5. 训练您的自定义 GRU 轨迹模型
```bash
cd "distance model/train"
pip install -r requirements.txt
# 运行训练脚本
python scripts/train.py --config configs/gru_baseline.yaml
```

---

## 项目开发与版本控制规范

本项目采用分支开发流程，禁止直接向 `main` 分支提交代码：
* **日常开发分支**：所有日常开发、Bug 修复和测试推送均提交至 **`development`** 分支。
* **正式发布**：当 `development` 经过集成测试达到稳定状态后，在 GitHub 网页端向 `main` 提交 **Pull Request** 进行合并。
* **版本标签**：合并后，在主分支上创建 Release 版本标签（例如 `v1.0.0`），以便后续回退或定位历史稳定版。

> 详细的 Git 命令行及 Pull Request 操作流程，请参见：**[Git 开发工作流指南 (GIT_WORKFLOW.md)](./GIT_WORKFLOW.md)**

---

## 项目规模

| 指标 | 数值 |
|------|------|
| Python文件数 | 128 |
| Python代码行数 | 15,798 |
| 总Episode数 | 444 |
| 模型文件数 | 20 |
| 总数据量 | ~3.4 GB |
| 飞行任务类型 | 13 种 |
| 距离范围 | 20m — 500m |

---

> 详细的系统架构分析，请参见：**[项目系统架构全景审计 (PROJECT_DOCUMENTATION.md)](./PROJECT_DOCUMENTATION.md)**
