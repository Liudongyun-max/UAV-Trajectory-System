# UAV Trajectory System — 完整项目说明文档

> **版本**: v7.0 | **更新日期**: 2026-06-18
> **项目规模**: 55,000+文件 | 128个Python文件 | 15,798行代码 | 444个Episode | 20个模型文件 | ~3.4GB数据

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构全景](#2-系统架构全景)
3. [核心模块深度剖析](#3-核心模块深度剖析)
4. [数据体系与样本结构](#4-数据体系与样本结构)
5. [YOLOv5 边缘端部署模块](#5-yolov5-边缘端部署模块)
6. [GRU 轨迹预测模型](#6-gru-轨迹预测模型)
7. [GRU 检测模块 (gru detect)](#7-gru-检测模块)
8. [YOLOv5+GRU 集成模块](#8-yolov5gru-集成模块)
9. [距离估算模块 (distance model)](#9-距离估算模块)
10. [训练框架模块](#10-训练框架模块)
11. [质量分级体系](#11-质量分级体系)
12. [技术栈与依赖](#12-技术栈与依赖)
13. [代码质量分析](#13-代码质量分析)
14. [部署与运维指南](#14-部署与运维指南)
15. [改进建议与路线图](#15-改进建议与路线图)

---

## 1. 项目概述

### 1.1 项目定位

**UAV Trajectory System** 是一套面向无人机 (UAV) 目标检测、轨迹跟踪、时序预测与距离估算的全栈系统，涵盖从仿真数据采集、数据质量审计、模型训练到边缘端实时部署的完整工作流程。

### 1.2 关键指标

| 指标 | 数值 |
|------|------|
| 总文件数 | **55,000+** |
| Python文件数 | **128** |
| Python代码行数 | **15,798** |
| 总Episode数 | **444** (run001: 39 + run002: 30 + run004: 375) |
| 飞行任务类型 | **13** 种 |
| 距离范围 | **20m — 500m** |
| 背景类型 | **3** 种 |
| 模型文件数 | **20** (.pth: 11, .pt: 5, .onnx: 2, .npz: 2) |
| 总数据量 | **~3.4 GB** |
| CSV数据文件 | **13,982** |
| PNG图片 | **22,762** |
| 视频文件 | **1,917** |
| YAML配置 | **3,069** |
| Markdown文档 | **907** |

---

## 2. 系统架构全景

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            UAV Trajectory System                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐              │
│  │  数据采集层        │    │  质量分析层        │    │  模型训练层        │              │
│  │  Gazebo + PX4     │───>│  analyze_episode  │───>│  train/          │              │
│  │  5路摄像头仿真      │    │  batch_analyze    │    │  GRU Training    │              │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘              │
│           │                       │                       │                         │
│           v                       v                       v                         │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐              │
│  │  export/input/    │    │  export/outputs/  │    │  models/         │              │
│  │  444 Episodes     │    │  Analysis Reports │    │  *.pth *.pt *.onnx│             │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘              │
│                                                             │                       │
│                                                             v                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                           推理与部署层                                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ gru detect/  │  │ yolov5+GRU/  │  │yolov5_model/ │  │distance model│   │   │
│  │  │ GRU独立推理   │  │ YOLO+GRU级联  │  │ RK3588边缘端  │  │ 距离估算      │   │   │
│  │  │ PyQt5 GUI    │  │ PyQt5 GUI    │  │ 5路实时检测    │  │ 多模式推理    │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
F:\UAV Trajectory System\
│
├── analyze_episode.py              # 核心8阶段分析引擎 (861行)
├── batch_analyze.py                # 批量分析调度器 (281行)
├── run_yolo_gru_test.py            # YOLO+GRU测试启动器 (26行)
│
├── yolov5_model/                   # RK3588边缘端部署模块 (1,344行)
│   ├── main.py                     # 多摄像头实时检测系统 (1,086行)
│   ├── yololib.py                  # RKNN YOLOv5推理封装 (197行)
│   ├── comms.py                    # UDP视频/数据传输 (61行)
│   ├── best include small.pt       # YOLOv5预训练权重 (13.79MB)
│   └── uav_yolo_rknn_report.md     # 系统审计报告
│
├── gru detect/                     # GRU独立检测模块 (1,319行)
│   ├── predict_gui.py              # PyQt5 GUI (389行)
│   ├── setup_env.bat               # 一键环境配置
│   ├── 启动UAV预测窗口.vbs          # 一键启动脚本
│   ├── src/
│   │   ├── model.py                # GRU模型定义 (116行)
│   │   ├── inference.py            # 推理会话管理 (335行)
│   │   ├── tracker.py              # 多目标特征追踪器 (467行)
│   │   └── paths.py                # 路径工具 (12行)
│   ├── models/                     # 预训练模型
│   │   ├── gru_baseline.pth        # 基线模型 (28KB)
│   │   └── gru_enhanced.pth        # 增强模型 (174KB)
│   └── outputs/                    # 推理输出 (21个episode)
│
├── yolov5+GRU/                     # YOLOv5+GRU端到端集成模块 (1,499行)
│   ├── pipeline.py                 # 级联检测+预测管线 (257行)
│   ├── detector.py                 # 多后端YOLOv5检测器 (238行)
│   ├── tracker.py                  # 单目标YOLO适配追踪器 (173行)
│   ├── model.py                    # GRU模型副本 (116行)
│   ├── gui_inference.py            # GUI推理会话 (193行)
│   ├── predict_gui.py              # 增强版PyQt5 GUI (444行)
│   ├── run.py                      # CLI运行器 (34行)
│   └── test_pipeline.py            # 诊断测试 (44行)
│
├── distance model/                 # 距离估算模块 (10,363行) [最大模块]
│   ├── distance_model_gui.py       # 距离模型可视化GUI (1,699行)
│   ├── segmentation_mode/          # 分割模式
│   │   ├── model.py                # GRU模型定义
│   │   ├── tracker.py              # 分割追踪器 (494行)
│   │   └── models/                 # 预训练模型
│   ├── yolo_mode/                  # YOLO模式
│   │   ├── detector.py             # YOLOv5检测器封装
│   │   └── models/                 # 预训练模型
│   ├── src/uav_distance_pipeline/  # 核心管线
│   ├── tools/                      # 工具脚本
│   ├── tests/                      # 测试文件 (15个)
│   ├── train/                      # 完整训练框架
│   │   ├── training/               # 训练模块
│   │   ├── scripts/                # 入口脚本
│   │   ├── configs/                # YAML配置
│   │   └── tests/                  # 测试 (10个)
│   ├── calibration/                # 摄像头标定
│   ├── dataset/                    # 数据集
│   ├── exports/                    # 导出数据
│   ├── input/                      # 输入数据 (run017 + run018, 20,820文件)
│   ├── models/                     # 训练模型
│   └── reports/                    # 分析报告
│
├── export/                         # 数据目录
│   ├── input/                      # 原始Episode数据 (444个)
│   │   ├── run001/                 # 39 episodes
│   │   ├── run002/                 # 30 episodes
│   │   └── run004/                 # 375 episodes
│   └── outputs/analysis/           # 分析输出
│
├── test  vedio/                    # 测试视频 (1,917个mp4+avi)
├── agent/skill/                    # AI Agent技能定义
├── distance/                       # 距离标定规范
├── document/                       # 技术文档
└── .github/                        # GitHub模板
```

---

## 3. 核心模块深度剖析

### 3.1 analyze_episode.py — 8阶段分析引擎

**位置**: `F:\UAV Trajectory System\analyze_episode.py`
**代码行数**: 861 行

#### 分析管线

| 阶段 | 功能 | 输出 |
|------|------|------|
| Stage 1 | 数据完整性检查 | integrity_report.json |
| Stage 2 | 时间同步分析 | sync_error_timeline.png, sync_frame_audit.csv |
| Stage 3 | 飞行阶段与可见性 | visibility_phase_audit.csv |
| Stage 4 | 世界运动与悬停质量 | trajectory_metrics.json |
| Stage 5 | 2D理想轨迹分析 | ideal_trajectory_xy.png |
| Stage 6 | 测量噪声分析 | noise_metrics.json |
| Stage 7 | 20+5滑动窗口审计 | window_audit.csv, valid/invalid_windows.csv |
| Stage 8 | 汇总JSON输出 | analysis_summary.json |

#### 数据流

```
输入: Episode目录
  │
  ├── episode.yaml ──────────> Stage 8 (元数据)
  ├── frames.csv ────────────> Stage 1,2,3,7,8
  ├── truth.csv ─────────────> Stage 1,2,3,4
  ├── ideal_tracks.csv ──────> Stage 1,5,6,7
  ├── measured_tracks.csv ───> Stage 1,6,7
  ├── events.jsonl ──────────> Stage 2 (事件解析)
  └── rgb.mp4 ───────────────> Stage 1 (帧数验证)
```

### 3.2 batch_analyze.py — 批量分析调度器

**位置**: `F:\UAV Trajectory System\batch_analyze.py`
**代码行数**: 281 行

**核心功能**:
- 并行分析（ThreadPoolExecutor，最多10个工作线程）
- 自动生成 Markdown 报告（analysis_report.md, recommendations.md）
- A/B/C/D 四级质量评定
- 批量汇总 JSON（batch_summary.json）

---

## 4. 数据体系与样本结构

### 4.1 数据规模

| Run | Episodes | 任务类型 | 距离范围 | 说明 |
|-----|----------|----------|----------|------|
| run001 | 39 | 13种 | 300-500m | 多任务多距离 |
| run002 | 30 | hover | 50-500m | 悬停距离梯度 |
| run004 | 375 | 多种 | 50-500m | 扩展数据集 |
| **总计** | **444** | | | |

### 4.2 13种飞行任务类型

| # | 任务类型 | 最大速度 | 最大加速度 | 描述 |
|---|----------|----------|------------|------|
| 1 | hover | 5 m/s | 2 m/s² | 定点悬停 |
| 2 | approaching | 10 m/s | 3 m/s² | 向观察者接近 |
| 3 | receding | 10 m/s | 3 m/s² | 远离观察者 |
| 4 | linear_left | 10 m/s | 3 m/s² | 向左线性飞行 |
| 5 | linear_right | 10 m/s | 3 m/s² | 向右线性飞行 |
| 6 | diagonal | 10 m/s | 3 m/s² | 对角线穿越 |
| 7 | fast_crossing | 15 m/s | 5 m/s² | 高速横向穿越 |
| 8 | accelerating | 12 m/s | 4 m/s² | 加速飞行 |
| 9 | decelerating | 12 m/s | 4 m/s² | 减速飞行 |
| 10 | sharp_turn | 10 m/s | 4 m/s² | 90度急转弯 |
| 11 | wide_turn | 10 m/s | 3 m/s² | 大弧度缓弯 |
| 12 | s_curve | 10 m/s | 4 m/s² | S形正弦曲线 |
| 13 | zigzag | 12 m/s | 5 m/s² | 快速锯齿形 |

### 4.3 数据统计

| 类型 | 数量 | 大小 |
|------|------|------|
| CSV数据文件 | 13,982 | 1,294 MB |
| PNG图片 | 22,762 | 691 MB |
| MP4视频 | 1,897 | 1,550 MB |
| AVI视频 | 20 | 477 MB |
| YAML配置 | 3,069 | 2.18 MB |
| JSON元数据 | 8,347 | 7.32 MB |
| **总计** | **~55,000** | **~3.4 GB** |

---

## 5. YOLOv5 边缘端部署模块

### 5.1 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| main.py | 1,086 | 多摄像头采集、运动检测、ROI提取、轨迹跟踪、推理调度、网络传输 |
| yololib.py | 197 | RKNN模型封装：模型加载、输入预处理、多分支YOLO输出解码、NMS后处理 |
| comms.py | 61 | UDP视频流和JSON数据遥测 |

### 5.2 硬件平台

| 硬件 | 接口 | 用途 |
|------|------|------|
| RK3588 NPU (3核) | rknnlite.api.RKNNLite | YOLOv5 推理（6 TOPS） |
| USB摄像头 (5路) | V4L2 (/dev/videoN) | 2560x1440 MJPG @15fps |
| 云台 | UDP 192.168.0.100:8888 | 目标坐标 + 距离 JSON |
| 地面站 | UDP 192.168.0.200:9999 | JPEG 视频流 |

### 5.3 TrajectoryFilter 多特征跟踪器

**匹配评分公式**:
```
match_score = 0.42 × distance_score
            + 0.23 × yolo_confidence
            + 0.17 × template_score
            + 0.18 × motion_score
```

---

## 6. GRU 轨迹预测模型

### 6.1 模型架构

```
输入: [Batch, 20, 8]
  │
  v
GRU Backbone: nn.GRU(input_size=8, hidden_size=32, num_layers=1, batch_first=True)
  │
  ├──> Classification Head: Linear(32→16) → ReLU → Linear(16→1) → sigmoid → UAV概率
  │
  └──> Prediction Head: Linear(32→32) → ReLU → Linear(32→10) → reshape → [B, 5, 2] 未来偏移
```

### 6.2 输入特征（8维）

| 特征 | 说明 |
|------|------|
| x_norm | 归一化X坐标 (x/2560) |
| y_norm | 归一化Y坐标 (y/1440) |
| relative_x | 相对首帧X位移 |
| relative_y | 相对首帧Y位移 |
| velocity_x | 帧间X速度 |
| velocity_y | 帧间Y速度 |
| acceleration_x | 帧间X加速度 |
| acceleration_y | 帧间Y加速度 |

---

## 7. GRU 检测模块 (gru detect)

### 7.1 核心组件

#### InferenceSession (inference.py, 335行)

- CSV数据驱动模式（含水平镜像校正）
- 纯视频驱动模式（自动目标检测）
- ADE（平均位移误差）实时计算
- 临时文件→验证→原子重命名的崩溃安全写入

#### FeatureTracker (tracker.py, 467行)

- 9步更新管线：检测→预测→匹配→惯性传播→去重→死锁检测→新轨迹创建
- CSV驱动偏置修正（跳跃门限过滤）
- 静态背景死锁检测（30/50帧跨度分析）
- Savitzky-Golay平滑 + 归一化特征提取

---

## 8. YOLOv5+GRU 集成模块

### 8.1 多后端检测器

| 后端 | 模型格式 | 适用环境 |
|------|----------|----------|
| RKNNImpl | .rknn | RK3588 NPU |
| ONNXImpl | .onnx | 通用PC |
| PyTorchImpl | .pt/.pth | PyTorch环境 |
| MockFallbackImpl | 无 | 降级检测 |

---

## 9. 距离估算模块 (distance model)

### 9.1 模块概述

**位置**: `F:\UAV Trajectory System\distance model\`
**代码行数**: 10,363 行（最大模块，占总代码65.6%）
**核心理念**: "物理基线 + ML残差修正 + GRU时序精化"

### 9.2 模块结构

```
distance model/
├── distance_model_gui.py       # 距离模型可视化GUI (1,699行)
├── segmentation_mode/          # 分割模式
│   ├── model.py                # GRU模型定义
│   ├── tracker.py              # 分割追踪器 (494行)
│   └── models/
│       └── gru_enhanced.pth    # 增强模型 (174KB)
├── yolo_mode/                  # YOLO模式
│   ├── detector.py             # YOLOv5检测器封装
│   └── models/
│       └── best include small.pt # YOLOv5权重 (13.79MB)
├── src/uav_distance_pipeline/  # 核心管线
├── tools/                      # 工具脚本
├── tests/                      # 测试文件 (15个)
├── train/                      # 完整训练框架
│   ├── training/               # 训练模块
│   ├── scripts/                # 入口脚本
│   ├── configs/                # YAML配置
│   └── tests/                  # 测试 (10个)
├── calibration/                # 摄像头标定
├── dataset/                    # 数据集
├── exports/                    # 导出数据
├── input/                      # 输入数据 (run017 + run018, 20,820文件)
├── models/                     # 训练模型
└── reports/                    # 分析报告
```

### 9.3 双模式检测架构

#### segmentation_mode (分割模式)

基于分割结果的轨迹跟踪与预测，使用GRU网络进行轨迹分类和未来位置预测。

#### yolo_mode (YOLO模式)

基于YOLOv5的目标检测与距离估算，支持实时视频流处理。

### 9.4 核心功能

- **Gazebo仿真**: 基于物理引擎的距离数据生成
- **数据集构建**: 多场景、多距离点的训练数据
- **MLP/GRU训练**: 距离估算模型训练
- **质量审计**: 特征泄漏检测、数据集划分验证
- **GRU输入导出**: 与主GRU模型的接口对接
- **可视化GUI**: 距离模型可视化界面
- **双模式推理**: segmentation_mode + yolo_mode

### 9.5 距离范围

| 指标 | 数值 |
|------|------|
| 距离范围 | 20m — 500m |
| 训练数据点 | 1,443 |
| 场景类型 | 3种 |
| 物理基线MAE | 0.86mm (仿真) |
| MLP最终MAE | 0.024mm (仿真) |

---

## 10. 训练框架模块

### 10.1 模块架构

```
train/
├── training/                   # 训练模块
│   ├── config/                 # 配置管理
│   ├── data/                   # 数据层
│   ├── models/                 # 模型层
│   ├── losses/                 # 损失函数
│   ├── trainers/               # 训练器
│   ├── evaluators/             # 评估器
│   ├── exporters/              # 模型导出
│   └── utils/                  # 工具函数
├── scripts/                    # 入口脚本
├── configs/                    # YAML配置
├── models/                     # 训练产物
└── tests/                      # 单元测试
```

### 10.2 训练配置

#### gru_baseline.yaml
```yaml
model:
  input_dim: 8
  hidden_dim: 32
  num_layers: 1
  future_steps: 5
  dropout: 0.1

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  cls_weight: 1.0
  reg_weight: 0.5
  patience: 15
```

---

## 11. 质量分级体系

### 11.1 四级评定标准

| 等级 | 条件 | 说明 |
|------|------|------|
| **A级** | HOVER阶段>200帧, valid_ratio>80%, sync<1ms | 优质训练数据 |
| **B级** | HOVER阶段>200帧, valid_ratio>50%, sync<2ms | 良好训练数据 |
| **C级** | 有效窗口>50个 | 可清洗后使用 |
| **D级** | 无效窗口=0, sync>10%无效 | 不可用于训练 |

---

## 12. 技术栈与依赖

### 12.1 核心依赖

| 类别 | 库 | 用途 |
|------|-----|------|
| 深度学习 | PyTorch | GRU模型训练与推理 |
| 计算机视觉 | OpenCV | 视频处理、图像处理、NMS |
| 数值计算 | NumPy, pandas | 数据处理 |
| 可视化 | matplotlib | 图表生成 |
| GUI | PyQt5 | 桌面界面 |
| 信号处理 | SciPy | Savitzky-Golay平滑 |
| NPU推理 | rknnlite | RK3588 NPU |
| 配置 | PyYAML | YAML配置解析 |
| 机器学习 | scikit-learn | 距离模型训练 |

### 12.2 模型文件清单

| 格式 | 数量 | 总大小 | 说明 |
|------|------|--------|------|
| .pth | 11 | 0.84 MB | PyTorch检查点 |
| .pt | 5 | 41.41 MB | PyTorch模型（含YOLOv5） |
| .onnx | 2 | 0.04 MB | ONNX导出 |
| .npz | 2 | 8.95 MB | NumPy模型 |
| **总计** | **20** | **51.24 MB** | |

---

## 13. 代码质量分析

### 13.1 代码统计

| 模块 | 文件数 | 行数 | 占比 |
|------|--------|------|------|
| 核心分析 | 2 | 1,142 | 7.2% |
| YOLOv5边缘端 | 3 | 1,344 | 8.5% |
| GRU检测 | 5 | 1,319 | 8.3% |
| YOLOv5+GRU集成 | 8 | 1,499 | 9.5% |
| 距离估算 | 108 | 10,363 | 65.6% |
| 其他 | 2 | 131 | 0.8% |
| **总计** | **128** | **15,798** | **100%** |

### 13.2 最大文件TOP10

| # | 文件 | 行数 |
|---|------|------|
| 1 | distance model/distance_model_gui.py | 1,699 |
| 2 | yolov5_model/main.py | 1,086 |
| 3 | analyze_episode.py | 861 |
| 4 | distance model/segmentation_mode/tracker.py | 494 |
| 5 | gru detect/src/tracker.py | 467 |
| 6 | yolov5+GRU/predict_gui.py | 444 |
| 7 | distance model/src/simulation_backend.py | 395 |
| 8 | gru detect/predict_gui.py | 389 |
| 9 | distance model/tools/train_distance_model.py | 359 |
| 10 | distance model/tools/audit_runtime_distance_pipeline.py | 228 |

### 13.3 架构亮点

1. **三份UAVTrajectoryNet**: 相同架构分布在多个模块中，保证导入隔离
2. **双追踪器实现**: 多目标（467行）vs 单目标（173行），适应不同场景
3. **四驱检测器**: RKNN/ONNX/PyTorch/Mock 自动降级
4. **双模式检测**: segmentation_mode + yolo_mode，灵活适配不同场景
5. **崩溃安全写入**: 临时文件→验证→原子重命名
6. **物理+ML混合架构**: 距离估算采用物理基线+MLP残差修正+GRU时序精化
7. **完整测试覆盖**: 距离模块包含25个单元测试

---

## 14. 部署与运维指南

### 14.1 GRU检测模块部署

```bash
# 一键配置
双击 setup_env.bat

# 或手动配置
conda create -n uav_gru python=3.10
conda activate uav_gru
cd "F:\UAV Trajectory System\gru detect"
pip install -r requirements.txt
python predict_gui.py
```

### 14.2 YOLOv5+GRU集成模块部署

```bash
cd "F:\UAV Trajectory System\yolov5+GRU"
conda activate uav_gru
python predict_gui.py
```

### 14.3 距离估算模块部署

```bash
cd "F:\UAV Trajectory System\distance model"
conda activate uav_gru
pip install -r train/requirements.txt

# 启动距离模型GUI
双击 双击启动测距可视化UI.bat

# 或命令行
python distance_model_gui.py
```

### 14.4 训练模块部署

```bash
cd "F:\UAV Trajectory System\distance model\train"
pip install -r requirements.txt
python scripts/train.py --config configs/gru_baseline.yaml
```

---

## 15. 改进建议与路线图

### 15.1 短期改进

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P0 | 统一模型副本 | 多份UAVTrajectoryNet合并为共享模块 |
| P0 | 添加requirements.txt | 项目根目录统一依赖声明 |
| P1 | 配置外部化 | 硬编码路径提取到配置文件 |
| P1 | 清理中间检查点 | 删除训练中间epoch文件 |

### 15.2 中期改进

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P1 | RKNN多进程重构 | multiprocessing替代threading绕过GIL |
| P2 | 增量学习支持 | 支持新数据微调现有模型 |
| P2 | Web界面 | 浏览器端实时预览 |

### 15.3 长期路线图

| 阶段 | 任务 | 预期效果 |
|------|------|----------|
| Phase 1 | 模型转换 ONNX→RKNN | 精度损失<1.5% |
| Phase 2 | Python多进程重构 | 5路摄像头10-15 FPS |
| Phase 3 | C++重写 (MPP+RGA+零拷贝) | >25 FPS, CPU<30% |

---

## 附录 A: 快速启动指南

### GRU检测（仅视频+GRU）

```bash
conda activate uav_gru
cd "F:\UAV Trajectory System\gru detect"
python predict_gui.py
# 选择视频 → 选择模型 → 点击启动
```

### YOLOv5+GRU（完整级联）

```bash
conda activate uav_gru
cd "F:\UAV Trajectory System\yolov5+GRU"
python predict_gui.py
# 选择视频 → 选择YOLO权重 → 选择GRU权重 → 调整置信度 → 点击启动
```

### 距离估算

```bash
conda activate uav_gru
cd "F:\UAV Trajectory System\distance model"
python distance_model_gui.py
# 启动距离模型可视化界面
```

### 训练新模型

```bash
conda activate uav_gru
cd "F:\UAV Trajectory System\distance model\train"
python scripts/train.py --config configs/gru_baseline.yaml
```

---

> **文档结束** — 此文档基于对项目全部128个Python源码文件、444个Episode、20个模型文件的深度分析自动生成。
