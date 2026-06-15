# UAV Trajectory System — 完整项目说明文档

> **版本**: v3.0 | **更新日期**: 2026-06-14 | **分析深度**: 全维度全模块深度剖析

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
9. [训练框架模块 (train)](#9-训练框架模块)
10. [质量分级体系](#10-质量分级体系)
11. [技术栈与依赖](#11-技术栈与依赖)
12. [代码质量分析](#12-代码质量分析)
13. [部署与运维指南](#13-部署与运维指南)
14. [改进建议与路线图](#14-改进建议与路线图)

---

## 1. 项目概述

### 1.1 项目定位

**UAV Trajectory System** 是一套面向无人机 (UAV) 目标检测、轨迹跟踪与时序预测的全栈系统，涵盖从仿真数据采集、数据质量审计、模型训练到边缘端实时部署的完整工作流程。

系统核心能力：
- **多摄像头实时检测**: 5路 USB 摄像头同时采集，RK3588 NPU 加速推理
- **运动引导级联检测**: 帧差分 + LCM 对比度滤波 + YOLO 精细识别
- **多目标轨迹跟踪**: Kalman 风格多特征跟踪器，支持模板匹配与动态门控
- **时序轨迹预测**: GRU 网络对 20 帧历史轨迹分类（真/伪目标）并预测未来 5 帧位置
- **端到端级联管线**: YOLOv5 + GRU 完整检测-跟踪-预测管线
- **自动化数据审计**: 8 阶段数据质量分析，A/B/C/D 四级评定

### 1.2 关键指标

| 指标 | 数值 |
|------|------|
| 总 Episode 数 | **444**（run001: 39 + run002: 30 + run004: 375） |
| 飞行任务类型 | **13** 种 |
| 距离范围 | **50m — 500m** |
| 背景类型 | **3** 种（empty_clean, flat_ground_mild, forest_edge_medium） |
| 每 Episode 文件数 | **17** 个 |
| Python 源码总量 | **~8,000+ 行** |
| GRU 模型参数量 | ~5K (基线) / ~25K (增强) |

---

## 2. 系统架构全景

### 2.1 系统架构图

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
│  │  444 Episodes     │    │  Analysis Reports │    │  *.pth           │          │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘          │
│                                                             │                   │
│                                                             v                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         推理与部署层                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │   │
│  │  │ gru detect/  │  │ yolov5+GRU/  │  │yolov5_model/ │                  │   │
│  │  │ GRU独立推理   │  │ YOLO+GRU级联  │  │ RK3588边缘端  │                  │   │
│  │  │ PyQt5 GUI    │  │ PyQt5 GUI    │  │ 5路实时检测    │                  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
F:\UAV Trajectory System\
│
├── analyze_episode.py              # 核心8阶段分析引擎 (953行)
├── batch_analyze.py                # 批量分析调度器 (329行)
├── run_yolo_gru_test.py            # YOLO+GRU测试启动器 (33行)
│
├── yolov5_model/                   # RK3588边缘端部署模块
│   ├── main.py                     # 多摄像头实时检测系统 (1173行)
│   ├── yololib.py                  # RKNN YOLOv5推理封装 (237行)
│   ├── comms.py                    # UDP视频/数据传输 (67行)
│   └── uav_yolo_rknn_report.md     # 系统审计报告 (239行)
│
├── gru detect/                     # GRU独立检测模块
│   ├── predict_gui.py              # PyQt5 GUI (437行)
│   ├── src/
│   │   ├── model.py                # GRU模型定义 (118行)
│   │   ├── inference.py            # 推理会话管理 (374行)
│   │   ├── tracker.py              # 多目标特征追踪器 (477行)
│   │   └── paths.py                # 路径工具 (18行)
│   ├── models/                     # 预训练模型
│   │   ├── gru_baseline.pth        # 基线模型 (0.03MB)
│   │   └── gru_enhanced.pth        # 增强模型 (0.17MB)
│   └── outputs/                    # 推理输出
│
├── yolov5+GRU/                     # YOLOv5+GRU端到端集成模块
│   ├── pipeline.py                 # 级联检测+预测管线 (272行)
│   ├── detector.py                 # 多后端YOLOv5检测器 (251行)
│   ├── tracker.py                  # 单目标YOLO适配追踪器 (186行)
│   ├── model.py                    # GRU模型副本 (118行)
│   ├── gui_inference.py            # GUI推理会话 (201行)
│   ├── predict_gui.py              # 增强版PyQt5 GUI (495行)
│   ├── run.py                      # CLI运行器 (37行)
│   └── test_pipeline.py            # 诊断测试 (48行)
│
├── train/                          # 模型训练框架
│   ├── training/
│   │   ├── config/                 # 配置管理
│   │   ├── data/                   # 数据集与加载器
│   │   ├── models/                 # 模型定义
│   │   ├── losses/                 # 损失函数
│   │   ├── trainers/               # 训练器与回调
│   │   ├── evaluators/             # 评估器
│   │   └── exporters/              # 模型导出
│   ├── scripts/                    # 入口脚本
│   ├── configs/                    # YAML配置
│   ├── models/                     # 训练产物
│   └── tests/                      # 单元测试
│
├── export/                         # 数据目录
│   ├── input/                      # 原始Episode数据
│   │   ├── run001/                 # 39 episodes
│   │   ├── run002/                 # 30 episodes
│   │   └── run004/                 # 375 episodes
│   └── outputs/analysis/           # 分析输出
│
├── agent/skill/                    # AI Agent技能定义
├── distance/                       # 距离标定规范
└── document/                       # 技术文档
```

---

## 3. 核心模块深度剖析

### 3.1 analyze_episode.py — 8阶段分析引擎

**位置**: `F:\UAV Trajectory System\analyze_episode.py`
**代码行数**: 953 行

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
输入 Episode 目录
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
**代码行数**: 329 行

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

### 4.3 Episode 数据结构（17个文件）

```
episode_hover_d300_vhover_empty_clean_seed0001/
├── camera_config.yaml          # 摄像头配置
├── camera_info.json            # 内参矩阵K, 畸变D, 投影P, HFOV
├── episode.yaml                # Episode完整元数据（30个字段）
├── events.jsonl                # 16行事件日志
├── execution.log               # 6行Shell命令日志
├── frames.csv                  # 逐帧: frame_id, 时间戳, 同步误差
├── ideal_tracks.csv            # 理想2D投影轨迹（GT）
├── measured_tracks.csv         # 模拟含噪检测
├── mission.yaml                # 飞行任务定义（17行）
├── preview/                    # 预览PNG帧
├── record_summary.json         # 录制摘要
├── recorder_ready.json         # 录制器就绪状态
├── rgb.mp4                     # 录制视频流
├── scenario.json               # 场景元数据
├── truth.csv                   # 3D世界状态
├── validation.json             # 预分析验证结果
└── world.world                 # Gazebo世界文件
```

---

## 5. YOLOv5 边缘端部署模块

### 5.1 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| main.py | 1173 | 多摄像头采集、运动检测、ROI提取、轨迹跟踪、推理调度、网络传输 |
| yololib.py | 237 | RKNN模型封装：模型加载、输入预处理、多分支YOLO输出解码、NMS后处理 |
| comms.py | 67 | UDP视频流和JSON数据遥测 |

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

**双路径确认机制**:
- YOLO直接确认: yolo_hits≥3, recent_hits≥3, misses=0, score≥0.40
- 标准确认: hits≥4, recent_hits≥3, misses≤2, score≥0.50, traj_score≥0.55

### 5.4 运动引导级联检测管线

```
摄像头 (2560x1440 MJPG @15fps)
  │
  v
[capture_job 线程] 每摄像头
  |-- 降采样至 1920x1080 灰度
  |-- absdiff 前帧差分
  |-- 阈值化 → 静态背景掩码 → 形态学清理
  |-- 轮廓提取 → 多级滤波管线
  |     |-- 面积/紧凑度/边缘密度检查
  |     |-- LCM 局部对比度滤波器
  |     |-- 灰度噪声/亮斑抑制
  |     |-- 垂直条带拒绝
  |     |-- 网格多样性配额
  |-- 评分排序 ROI (每帧最多6个)
  |-- 轨迹种子搜索 ROI (最多3个)
  |-- 创建 640x640 融合画布
  │
  v (inf_queues)
[inference_worker 线程] ── 单 YoloRKNN 实例
  |-- rknn.inference() 使用全部3核NPU
  |-- 解码YOLO输出 + NMS
  │
  v (res_queues)
[capture_job 线程] ── 结果处理
  |-- 坐标映射 + TrajectoryFilter.update()
  |-- 确认目标 → UDP发送至云台
  |-- 绘制叠加 → UDP发送视频
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

**可选12维版本**: 增加 w_norm, h_norm, aspect_ratio, detector_confidence

### 6.3 输出

- **分类logit**: [B, 1] → sigmoid → UAV概率 (0.0~1.0)
- **未来偏移**: [B, 5, 2] → 未来5帧相对位置偏移

### 6.4 预训练模型

| 模型 | 路径 | 参数量 | 大小 |
|------|------|--------|------|
| 基线模型 | gru detect/models/gru_baseline.pth | ~5K | 0.03MB |
| 增强模型 | gru detect/models/gru_enhanced.pth | ~25K | 0.17MB |

---

## 7. GRU 检测模块 (gru detect)

### 7.1 模块架构

```
gru detect/
├── predict_gui.py              # PyQt5 GUI 入口
├── src/
│   ├── model.py                # UAVTrajectoryNet 模型定义
│   ├── inference.py            # InferenceSession 推理会话
│   ├── tracker.py              # FeatureTracker 多目标追踪器
│   └── paths.py                # 路径工具函数
├── models/                     # 预训练模型
└── outputs/                    # 推理输出
```

### 7.2 核心组件

#### InferenceSession (inference.py, 374行)

管理视频读取、模型推理、视频写入的完整生命周期：

- `_open()`: 初始化模型、视频、追踪器
- `step()`: 处理单帧，返回渲染结果
- `_render_frame()`: GRU推理 + 可视化渲染
- `close()`: 保存输出并清理资源

**特性**:
- CSV数据驱动模式（含水平镜像校正）
- 纯视频驱动模式（自动目标检测）
- ADE（平均位移误差）实时计算
- 临时文件→验证→原子重命名的崩溃安全写入

#### FeatureTracker (tracker.py, 477行)

**多目标追踪器**，支持：

- 9步更新管线：检测→预测→匹配→惯性传播→去重→死锁检测→新轨迹创建
- CSV驱动偏置修正（跳跃门限过滤）
- 静态背景死锁检测（30/50帧跨度分析）
- Savitzky-Golay平滑 + 归一化特征提取

### 7.3 启动方式

```bash
# 方式一：双击启动脚本
启动UAV预测窗口.vbs

# 方式二：命令行
conda activate uav_gru
python predict_gui.py
```

---

## 8. YOLOv5+GRU 集成模块

### 8.1 模块架构

```
yolov5+GRU/
├── pipeline.py                 # YoloGRUPipeline 级联管线核心
├── detector.py                 # 多后端YOLOv5检测器
│   ├── YoloDetector            # 抽象基类
│   ├── RKNNImpl                # RK3588 NPU推理
│   ├── ONNXImpl                # ONNX Runtime推理
│   ├── PyTorchImpl             # PyTorch推理
│   └── MockFallbackImpl        # 零模型降级检测
├── tracker.py                  # 单目标YOLO适配追踪器
├── model.py                    # GRU模型（副本）
├── gui_inference.py            # GUI推理会话
├── predict_gui.py              # 增强版PyQt5 GUI
├── run.py                      # CLI运行器
└── test_pipeline.py            # 诊断测试
```

### 8.2 YoloGRUPipeline 核心管线

**级联检测策略**:

1. **追踪模式**（有历史）: 以last_pos为中心裁剪640x640 ROI → YOLO检测
2. **全局捕获模式**（无历史）:
   - 轨A: 全图YOLO检测
   - 轨B: MockFallback亮点定位 → 256x256 ROI → YOLO精细检测

**处理流程**:
```
视频帧 → YOLO检测(ROI/全局) → FeatureTracker关联 → GRU预测 → 渲染输出
```

### 8.3 多后端检测器

| 后端 | 模型格式 | 适用环境 |
|------|----------|----------|
| RKNNImpl | .rknn | RK3588 NPU |
| ONNXImpl | .onnx | 通用PC |
| PyTorchImpl | .pt/.pth | PyTorch环境 |
| MockFallbackImpl | 无 | 降级检测 |

**工厂函数** `get_detector()`: 根据模型文件扩展名自动选择后端

### 8.4 增强版GUI (predict_gui.py, 495行)

**新增功能**:
- YOLO模型权重选择（.pt/.onnx/.rknn）
- 置信度阈值滑块（10-90%实时调整）
- show_box / show_history / show_pred_red_blue 开关
- 自动扫描yolov5_model/和detect/models/目录

---

## 9. 训练框架模块 (train)

### 9.1 模块架构

```
train/
├── training/
│   ├── config/                 # 配置管理 (experiment.py)
│   ├── data/                   # 数据层
│   │   ├── dataset.py          # UAVTrajectoryDataset
│   │   ├── dataloader.py       # DataLoader封装
│   │   ├── splitter.py         # 数据集划分
│   │   ├── negative_generator.py # 负样本合成
│   │   └── transforms.py       # 数据变换
│   ├── models/                 # 模型层
│   │   ├── gru_trajectory.py   # UAVTrajectoryNet
│   │   └── registry.py         # 模型注册表
│   ├── losses/                 # 损失函数
│   │   ├── combined_loss.py    # 分类+回归组合损失
│   │   └── focal_loss.py       # Focal Loss
│   ├── trainers/               # 训练器
│   │   ├── base_trainer.py     # 基类
│   │   ├── gru_trainer.py      # GRU训练器
│   │   └── callbacks/          # 回调函数
│   ├── evaluators/             # 评估器
│   ├── exporters/              # 模型导出 (ONNX/TorchScript)
│   └── utils/                  # 工具函数
├── scripts/                    # 入口脚本
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 评估入口
│   └── export_model.py         # 导出入口
├── configs/                    # YAML配置
│   ├── gru_baseline.yaml       # 基线配置
│   └── gru_enhanced.yaml       # 增强配置
├── models/                     # 训练产物
│   ├── gru_baseline_final.pth  # 基线模型
│   └── gru_enhanced_final.pth  # 增强模型
└── tests/                      # 单元测试
```

### 9.2 训练配置

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

#### gru_enhanced.yaml
```yaml
model:
  input_dim: 12
  hidden_dim: 64
  num_layers: 2
  future_steps: 5
  dropout: 0.2

training:
  batch_size: 32
  learning_rate: 0.0005
  epochs: 200
  patience: 20
```

### 9.3 训练入口

```bash
cd train
python scripts/train.py --config configs/gru_baseline.yaml
python scripts/evaluate.py --model models/gru_baseline_final.pth --data_dir ../export/input --visualize
python scripts/export_model.py --model models/gru_baseline_final.pth --format onnx
```

---

## 10. 质量分级体系

### 10.1 四级评定标准

| 等级 | 条件 | 说明 |
|------|------|------|
| **A级** | HOVER阶段>200帧, valid_ratio>80%, sync<1ms | 优质训练数据 |
| **B级** | HOVER阶段>200帧, valid_ratio>50%, sync<2ms | 良好训练数据 |
| **C级** | 有效窗口>50个 | 可清洗后使用 |
| **D级** | 无效窗口=0, sync>10%无效 | 不可用于训练 |

---

## 11. 技术栈与依赖

### 11.1 核心依赖

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

### 11.2 模型文件清单

| 文件 | 大小 | 位置 |
|------|------|------|
| gru_baseline.pth | 0.03MB | gru detect/models/ |
| gru_enhanced.pth | 0.17MB | gru detect/models/ |
| gru_baseline_final.pth | 0.03MB | train/models/ |
| gru_enhanced_final.pth | 0.17MB | train/models/ |
| 训练检查点 | 多个 | train/logs/ |

---

## 12. 代码质量分析

### 12.1 代码统计

| 模块 | 文件数 | 总行数 |
|------|--------|--------|
| 核心分析 | 2 | 1,282 |
| YOLOv5边缘端 | 3 | 1,477 |
| GRU检测 | 5 | 1,424 |
| YOLOv5+GRU集成 | 8 | 1,608 |
| 训练框架 | 20+ | 2,000+ |
| **总计** | **38+** | **~8,000+** |

### 12.2 架构亮点

1. **三份UAVTrajectoryNet**: 相同架构分布在三个模块中，保证导入隔离
2. **双追踪器实现**: 多目标（477行）vs 单目标（186行），适应不同场景
3. **级联检测策略**: YOLO全局→Mock降级→ROI精细，确保远距离目标捕获
4. **崩溃安全写入**: 临时文件→验证→原子重命名
5. **多后端检测器**: RKNN/ONNX/PyTorch/Mock 自动降级

---

## 13. 部署与运维指南

### 13.1 GRU检测模块部署

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

### 13.2 YOLOv5+GRU集成模块部署

```bash
cd "F:\UAV Trajectory System\yolov5+GRU"
conda activate uav_gru
pip install -r ../gru\ detect/requirements.txt

# GUI启动
python predict_gui.py

# CLI启动
python run.py --video input.mp4 --output output.avi
```

### 13.3 训练模块部署

```bash
cd "F:\UAV Trajectory System\train"
pip install -r requirements.txt

# 训练
python scripts/train.py --config configs/gru_baseline.yaml

# 评估
python scripts/evaluate.py --model models/gru_baseline_final.pth --data_dir ../export/input

# 导出
python scripts/export_model.py --model models/gru_baseline_final.pth --format onnx
```

### 13.4 RK3588边缘端部署

```bash
cd yolov5_model
python main.py
```

---

## 14. 改进建议与路线图

### 14.1 短期改进

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P0 | 统一模型副本 | 三份UAVTrajectoryNet合并为一个共享模块 |
| P0 | 添加requirements.txt | 项目根目录统一依赖声明 |
| P1 | 配置外部化 | 硬编码路径提取到配置文件 |
| P1 | 日志系统 | 添加结构化日志替换print输出 |

### 14.2 中期改进

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P1 | RKNN多进程重构 | multiprocessing替代threading绕过GIL |
| P2 | 增量学习支持 | 支持新数据微调现有模型 |
| P2 | Web界面 | 浏览器端实时预览 |

### 14.3 长期路线图

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

### 训练新模型

```bash
conda activate uav_gru
cd "F:\UAV Trajectory System\train"
python scripts/train.py --config configs/gru_baseline.yaml
```

---

> **文档结束** — 此文档基于对项目全部源码、文档和数据的深度分析自动生成。
