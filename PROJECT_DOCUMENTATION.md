# UAV Trajectory System — 完整项目说明文档

> **文档版本**: v2.0 | **生成日期**: 2026-06-13 | **分析深度**: 全维度深度剖析

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构全景](#2-系统架构全景)
3. [核心模块深度剖析](#3-核心模块深度剖析)
4. [数据体系与样本结构](#4-数据体系与样本结构)
5. [分析引擎详解](#5-分析引擎详解)
6. [边缘端部署模块](#6-边缘端部署模块)
7. [GRU 轨迹预测模型](#7-gru-轨迹预测模型)
8. [距离估算系统](#8-距离估算系统)
9. [质量分级体系](#9-质量分级体系)
10. [技术栈与依赖](#10-技术栈与依赖)
11. [代码质量分析](#11-代码质量分析)
12. [部署与运维指南](#12-部署与运维指南)
13. [改进建议与路线图](#13-改进建议与路线图)

---

## 1. 项目概述

### 1.1 项目定位

**UAV Trajectory System** 是一套面向无人机 (UAV) 目标检测与轨迹分析的全栈系统，涵盖从仿真数据采集、数据质量审计、模型训练到边缘端实时部署的完整工作流程。系统以 **PX4 SITL + Gazebo** 仿真环境为基础，通过多摄像头采集无人机飞行数据，经过严格的六阶段质量分析后，训练 GRU 时序轨迹网络，并最终部署到 **RK3588 边缘计算平台**上进行实时目标检测与跟踪。

### 1.2 核心目标

| 目标 | 描述 |
|------|------|
| **数据质量保障** | 通过自动化审计管线，对每一帧无人机轨迹数据进行完整性、同步性、可见性、运动质量、测量噪声和训练窗口有效性六维度评估 |
| **多任务覆盖** | 支持 13 种飞行任务类型（悬停、接近、远离、线性飞行、急转弯等），覆盖无人机常见机动模式 |
| **边缘端实时检测** | 在 RK3588 NPU 上运行 YOLOv5s 模型，同时处理 5 路摄像头，实现运动引导的级联检测 |
| **轨迹时序预测** | 通过 GRU 网络对 20 帧历史轨迹进行分类（真/伪目标）和未来 5 帧位置预测，支撑云台前馈控制 |

### 1.3 关键指标

| 指标 | 数值 |
|------|------|
| 总 Episode 数 | **69**（run001: 39 + run002: 30） |
| 飞行任务类型 | **13** 种 |
| 距离范围 | **50m — 500m** |
| 背景类型 | **3** 种（empty_clean, flat_ground_mild, forest_edge_medium） |
| 每 Episode 文件数 | **17** 个 |
| 每 Episode 分析输出 | **28** 个文件（JSON + CSV + 19 PNG 图表） |
| 总分析报告数 | **138** 个 Markdown 文件 |
| Python 源码总量 | **~2,748 行** |
| 文档总量 | **~1,984 行** |

---

## 2. 系统架构全景

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UAV Trajectory System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │  数据采集层        │    │  质量分析层        │    │  模型训练层        │       │
│  │  Gazebo + PX4     │───>│  analyze_episode  │───>│  GRU Training    │       │
│  │  5路摄像头仿真      │    │  batch_analyze    │    │  (外部脚本)       │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│           │                       │                       │                  │
│           │                       │                       │                  │
│           v                       v                       v                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │  export/input/    │    │  export/outputs/  │    │  model/          │       │
│  │  69 Episodes      │    │  138 MD Reports   │    │  yolov5s.rknn    │       │
│  │  17 files each    │    │  1314 PNG Charts  │    │  gru_model.pt    │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│                                                             │                │
│                                                             v                │
│                                                    ┌──────────────────┐      │
│                                                    │  边缘端部署层      │      │
│                                                    │  RK3588 NPU      │      │
│                                                    │  5路实时检测       │      │
│                                                    │  轨迹跟踪 + 云台   │      │
│                                                    └──────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块关系图

```
batch_analyze.py (批量调度器)
    │
    │  subprocess.run() 逐 Episode 调用
    │
    v
analyze_episode.py (单 Episode 分析引擎)
    │
    │  读取: CSV, JSON, YAML, MP4
    │  写入: JSON, CSV, PNG
    │
    v
batch_analyze.py (汇总结果)
    │
    │  读取 JSON/CSV 输出
    │  生成: Markdown 报告, batch_summary.json
    │
    v
最终输出:
  ├── 每 Episode: analysis_report.md, recommendations.md, figures/*.png, *.json, *.csv
  └── 每 Run: batch_summary.json


yolov5_model/ (边缘端检测模块)
    │
    ├── main.py ──── 多摄像头采集 + 运动检测 + 轨迹跟踪
    │     │
    │     ├── yololib.py ──── RKNN 推理封装
    │     └── comms.py ──── UDP 视频/数据传输
    │
    └── 使用: yolov5s.rknn 模型
```

### 2.3 目录结构

```
F:\UAV Trajectory System\
├── analyze_episode.py              # 核心单 Episode 分析引擎 (953 行)
├── batch_analyze.py                # 批量分析调度器 (318 行)
├── agent/
│   └── skill/
│       └── uav_data_analysis_skill.md  # AI Agent 分析技能定义 (138 行)
├── distance/
│   └── distance_scale_specification.md # 距离/尺度标定规范 (184 行)
├── document/
│   └── GRU_无人机时序轨迹网络详细说明.md  # GRU 模型技术文档 (1662 行)
├── export/
│   ├── input/                     # 原始 Episode 数据
│   │   ├── run001/                # 39 episodes (13 类型 x 3 距离)
│   │   └── run002/                # 30 episodes (hover x 10 距离 x 3 背景)
│   └── outputs/analysis/          # 分析输出
│       ├── run001/                # 39 episode 分析结果
│       └── run002/                # 30 episode 分析结果
└── yolov5_model/
    ├── main.py                    # YOLOv5 多摄像头检测管线 (1173 行)
    ├── yololib.py                 # RKNN 推理封装 (237 行)
    ├── comms.py                   # UDP 视频/数据传输 (67 行)
    └── uav_yolo_rknn_report.md    # YOLOv5-RKNN 系统审计报告 (239 行)
```

---

## 3. 核心模块深度剖析

### 3.1 analyze_episode.py — 单 Episode 分析引擎

**位置**: `F:\UAV Trajectory System\analyze_episode.py`
**代码行数**: 953 行
**职责**: 对单个 Episode 进行完整的七阶段数据质量分析

#### 3.1.1 导入与全局常量

| 模块 | 用途 |
|------|------|
| `os` | 路径操作、目录创建 |
| `json` | JSON 报告读写 |
| `argparse` | CLI 参数解析 |
| `numpy` | 数值运算（sqrt, inf 检查, 分位数, 弧度转换） |
| `pandas` | CSV DataFrame 操作 |
| `matplotlib` | 非交互式后端 (`Agg`)，生成 13+ 张图表 |
| `cv2` | 视频帧数验证（`VideoCapture`） |
| `yaml`（延迟导入） | 读取 `episode.yaml` 元数据 |

#### 3.1.2 全局配置常量

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `required_files` | 12 个文件名 | Episode 必需文件清单 |
| `sim_arm_t` | 6.457 | ARM 事件模拟时间 |
| `sim_exec_t` | 36.492 | EXECUTE_MANEUVER 模拟时间 |
| `sim_post_t` | 66.614 | POST_ROLL 模拟时间 |
| `sim_land_t` | 69.622 | LAND 模拟时间 |
| `sim_finish_t` | 74.639 | FINISHED 模拟时间 |

#### 3.1.3 七阶段分析管线

##### Stage 1: 数据完整性分析 (lines 62-182)

**功能**: 验证 Episode 数据文件的完整性和一致性

**检查项目**:
1. 12 个必需文件的存在性与大小
2. 四个核心 CSV（frames.csv, truth.csv, ideal_tracks.csv, measured_tracks.csv）的加载
3. CSV 行数一致性检查
4. frame_id 连续性验证（从首值开始的连续整数序列）
5. 所有数值列的 NaN/Inf 检查
6. `image_timestamp` 和 `state_timestamp` 的单调递增检查
7. 所有 CSV 的 frame_id 对齐验证
8. `rgb.mp4` 视频帧数与 frames.csv 行数匹配验证

**输出**: `integrity_report.json`

##### Stage 2: 时间同步分析 (lines 186-339)

**功能**: 评估图像采集与状态记录之间的时间同步质量

**关键逻辑**:
- 逐行读取 `events.jsonl`，动态计算 `t_offset`（wall_time - sim_timestamp）
- 支持 `MAVROS_*` 和短状态名两种事件格式
- 动态计算 `sim_settle_t`：找到无人机达到最大高度 97.5% 的首个时刻
- 在五个帧子集上计算同步指标：全部帧、去除前15帧、去除前1秒、稳定悬停期、仅可见帧

**指标**: mean, median, P90, P95, P99, max, invalid_ratio

**异常审计**: 识别 `sync_error_ms > 500` 的帧，导出至 `sync_frame_audit.csv`

**输出**: `sync_error_timeline.png`, `sync_error_histogram.png`, `sync_frame_audit.csv`

##### Stage 3: 飞行阶段与可见性分析 (lines 342-483)

**功能**: 将每帧分类到飞行阶段，并分析目标可见性

**飞行阶段分类** (`get_flight_phase()`):

| 阶段 | 判定条件 |
|------|----------|
| PRE_ROLL | t < sim_arm_t |
| ARM | t ∈ [sim_arm_t, sim_settle_t) |
| TAKEOFF | z > 0.1m 且未稳定 |
| SETTLE | 3秒稳定窗口内，abs(vz) >= 0.1 |
| HOVER | z > 0.1m 且稳定 |
| POST_ROLL | t ∈ [sim_post_t, sim_land_t) |
| LAND | t ∈ [sim_land_t, sim_finish_t) |
| UNKNOWN | 其他 |

**输出**: `visibility_phase_audit.csv`, `phase_visibility_timeline.png`, `visibility_timeline.png`, `image_position_timeline.png`

##### Stage 4: 世界运动与悬停质量分析 (lines 486-583)

**功能**: 评估无人机在世界坐标系中的运动质量和悬停稳定性

**悬停质量指标**:
- 水平 RMS 漂移
- P95 半径
- 最大漂移距离
- 高度标准差
- 最大高度偏差
- 平均/P95 水平速度
- 净位移

**轨迹指标**: 世界坐标范围、最大速度大小、最大加速度大小

**输出**: `trajectory_metrics.json`, `world_position.png`, `world_velocity.png`, `attitude.png`, `hover_xy_scatter.png`

##### Stage 5: 2D 理想轨迹分析 (lines 586-643)

**功能**: 分析图像平面上的 2D 投影轨迹

**计算内容**: 像素速度（dt, pixel_vx, pixel_vy, pixel_v）

**输出**: `ideal_trajectory_xy.png`, `ideal_xy_timeline.png`, `bbox_size_timeline.png`, `pixel_velocity.png`

##### Stage 6: 测量噪声分析 (lines 646-781)

**功能**: 量化理想轨迹与模拟检测之间的误差

**分析内容**:
- 中心平移误差（欧氏距离）
- 边界框宽/高误差
- X/Y 偏差
- 缺失检测统计（连续缺失运行分布）
- ID 切换计数
- 3-sigma 异常值检测
- 置信度统计（mean/P95/max）

**输出**: `noise_metrics.json`, `ideal_measured_overlay.png`, `center_error_timeline.png`, `center_error_histogram.png`, `error_xy_scatter.png`, `confidence_timeline.png`, `missing_timeline.png`

##### Stage 7: 20+5 滑动窗口审计 (lines 784-912)

**功能**: 评估训练数据窗口的有效性（20帧历史 + 5帧未来）

**八项有效性检查**:

| 检查项 | 说明 |
|--------|------|
| INVALID_TIMESTAMP | 时间戳连续性（恰好25帧连续） |
| INVALID_TRACK_SWITCH | 单一 track ID（无轨迹切换） |
| INVALID_PHASE_CROSSING | 无起飞前/着陆后阶段 |
| INVALID_SYNC | 同步有效率 >= 95% |
| INVALID_NO_FUTURE_GT | 未来5帧有理想GT |
| INVALID_INVISIBLE | 未来5帧可见且同步有效 |
| INVALID_LONG_MISSING | 历史20帧：至少18个有效测量，最大连续缺失 <= 2 |

**状态码**: VALID, INVALID_TIMESTAMP, INVALID_TRACK_SWITCH, INVALID_PHASE_CROSSING, INVALID_SYNC, INVALID_NO_FUTURE_GT, INVALID_INVISIBLE, INVALID_LONG_MISSING

**跨步长比较**: stride 1/3/5 的有效窗口计数

**输出**: `window_audit.csv`, `valid_windows.csv`, `invalid_windows.csv`

##### Stage 8: 汇总 JSON 输出 (lines 915-953)

**功能**: 从 `episode.yaml` 读取元数据，组装所有阶段结果为最终汇总

**输出**: `analysis_summary.json`

#### 3.1.4 每 Episode 输出文件清单（共 28 个）

| 文件 | 来源阶段 |
|------|----------|
| `integrity_report.json` | Stage 1 |
| `sync_frame_audit.csv` | Stage 2（条件） |
| `sync_error_timeline.png` | Stage 2 |
| `sync_error_histogram.png` | Stage 2 |
| `visibility_phase_audit.csv` | Stage 3 |
| `phase_visibility_timeline.png` | Stage 3 |
| `visibility_timeline.png` | Stage 3 |
| `image_position_timeline.png` | Stage 3 |
| `trajectory_metrics.json` | Stage 4 |
| `world_position.png` | Stage 4 |
| `world_velocity.png` | Stage 4 |
| `attitude.png` | Stage 4 |
| `hover_xy_scatter.png` | Stage 4 |
| `ideal_trajectory_xy.png` | Stage 5 |
| `ideal_xy_timeline.png` | Stage 5 |
| `bbox_size_timeline.png` | Stage 5 |
| `pixel_velocity.png` | Stage 5 |
| `noise_metrics.json` | Stage 6 |
| `ideal_measured_overlay.png` | Stage 6 |
| `center_error_timeline.png` | Stage 6 |
| `center_error_histogram.png` | Stage 6 |
| `error_xy_scatter.png` | Stage 6 |
| `confidence_timeline.png` | Stage 6 |
| `missing_timeline.png` | Stage 6 |
| `window_audit.csv` | Stage 7 |
| `valid_windows.csv` | Stage 7 |
| `invalid_windows.csv` | Stage 7 |
| `analysis_summary.json` | Stage 8 |

---

### 3.2 batch_analyze.py — 批量分析调度器

**位置**: `F:\UAV Trajectory System\batch_analyze.py`
**代码行数**: 318 行
**职责**: 批量发现 Episode，调用分析引擎，聚合结果，生成 Markdown 报告

#### 3.2.1 全局常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `INPUT_ROOT` | `F:\UAV Trajectory System\export\input` | 输入根目录 |
| `OUTPUT_ROOT` | `F:\UAV Trajectory System\export\outputs` | 输出根目录 |
| `ANALYZE_SCRIPT` | `F:\UAV Trajectory System\analyze_episode.py` | 分析脚本路径 |
| `PYTHON_EXE` | `D:\anaconda3\python.exe` | Python 解释器路径（硬编码） |

#### 3.2.2 核心函数 `analyze_run(run_name)`

**执行流程**:

1. **Episode 发现**: 遍历 `INPUT_ROOT/<run_name>/` 下的机动类型目录和 Episode 子目录
2. **子进程执行**: 对每个 Episode 调用 `subprocess.run([PYTHON_EXE, ANALYZE_SCRIPT, "-i", input_dir])`
3. **结果读取**: 读取 `analysis_summary.json`, `noise_metrics.json`, `trajectory_metrics.json`, `visibility_phase_audit.csv`
4. **质量分级**: 四级评定（A/B/C/D）
5. **任务适配文本**: 根据 `meta_mission` 类型调整报告章节标题
6. **Markdown 生成**: 为每个 Episode 生成 `analysis_report.md` 和 `recommendations.md`
7. **批量汇总**: 写入 `batch_summary.json`

#### 3.2.3 CLI 参数

| 参数 | 标志 | 必需 | 说明 |
|------|------|------|------|
| `--run` | `-r` | 否 | 指定处理的 run 名称；省略则处理所有 run* 目录 |

---

## 4. 数据体系与样本结构

### 4.1 Episode 数据架构

每个 Episode 包含 **17 个文件**，构成完整的无人机飞行数据记录：

```
episode_hover_d300_vhover_empty_clean_seed0001/
├── camera_config.yaml          # 摄像头配置（debug/production/physical_2k）
├── camera_info.json            # 内参矩阵K, 畸变D, 投影P, HFOV
├── episode.yaml                # Episode 完整元数据（30个字段）
├── events.jsonl                # 16行事件日志
├── execution.log               # 6行Shell命令日志
├── frames.csv                  # 逐帧: frame_id, 时间戳, 同步误差
├── ideal_tracks.csv            # 理想2D投影轨迹
├── measured_tracks.csv         # 模拟含噪检测
├── mission.yaml                # 飞行任务定义（17行）
├── preview/                    # 预览PNG帧（每30帧）
├── record_summary.json         # {episode_id, frames, visible_frames}
├── recorder_ready.json         # 录制器就绪状态
├── rgb.mp4                     # 录制视频流
├── scenario.json               # 场景元数据
├── truth.csv                   # 3D世界状态
├── validation.json             # 预分析验证结果
└── world.world                 # Gazebo世界文件
```

### 4.2 episode.yaml 元数据（30 个字段）

```yaml
episode_id: episode_hover_d300_vhover_empty_clean_seed0001
capture_mode: gazebo_live          # 采集管线模式
fallback_used: false               # 是否使用mavros回退
training_eligible: true            # 预筛选合格标志
distance_m: 300.0                  # 观察者到目标距离（米）
mission: hover                     # 飞行任务类型
noise_profile: clean               # 噪声: clean | mild | medium
background: empty                  # 背景: empty | flat_ground | forest_edge
camera_profile: debug              # 摄像头: debug(1280x720@15fps) | production | physical_2k
random_seed: 1                     # 仿真随机种子
frame_count: 1110                  # 总帧数
valid_frame_count: 905             # 同步有效帧数
quality_grade: A                   # 预计算质量等级
# ... 其余字段包含 SHA256 哈希、路径信息等
```

### 4.3 mission.yaml 任务定义

```yaml
name: hover
warmup_s: 5                         # 预热时长
duration_s: 15                      # 任务执行时长
altitude_m: 40                      # 目标飞行高度
max_speed_mps: 5                    # 最大允许速度
max_acceleration_mps2: 2            # 最大允许加速度
post_roll_s: 3                      # 后录制时长
settle_s: 3                         # 稳定阶段时长
waypoints_ned:                      # NED航点列表
  - [0, 0, -40]
  - [0, 0, -40]
safety_bounds:                      # 安全约束
  x_abs_max_m: 120
  y_abs_max_m: 120
  z_min_m: 5
  z_max_m: 80
```

### 4.4 核心 CSV 数据格式

#### frames.csv
| 字段 | 类型 | 说明 |
|------|------|------|
| frame_id | int | 帧序号 |
| image_timestamp | float | 图像时间戳 |
| state_timestamp | float | 状态时间戳 |
| sync_error_ms | float | 同步误差（毫秒） |
| sync_valid | bool | 同步是否有效 |

#### truth.csv
| 字段 | 类型 | 说明 |
|------|------|------|
| frame_id | int | 帧序号 |
| world_x/y/z | float | 世界坐标位置 |
| velocity_x/y/z | float | 世界坐标速度 |
| acceleration_x/y/z | float | 世界坐标加速度 |
| roll/pitch/yaw | float | 姿态角（弧度） |
| visible | bool | 目标是否可见 |
| range_m | float | 距离（米） |
| projected_u/v | float | 投影像素坐标 |

#### ideal_tracks.csv
| 字段 | 类型 | 说明 |
|------|------|------|
| frame_id | int | 帧序号 |
| x_center/y_center | float | 中心像素坐标 |
| bbox_width/bbox_height | float | 边界框尺寸 |
| visible | bool | 是否可见 |
| track_id | int | 轨迹ID |

#### measured_tracks.csv
| 字段 | 类型 | 说明 |
|------|------|------|
| frame_id | int | 帧序号 |
| x_center/y_center | float | 中心像素坐标 |
| bbox_width/bbox_height | float | 边界框尺寸 |
| detector_confidence | float | 检测置信度 |
| is_missing | bool | 是否缺失 |
| track_id | int | 轨迹ID |

### 4.5 数据规模统计

#### run001: 多任务多距离数据集

| 维度 | 值 |
|------|-----|
| Episode 总数 | 39 |
| 任务类型 | 13 种 |
| 距离 | 300m, 400m, 500m |
| 每距离背景 | 1 种（empty_clean, flat_ground_mild, forest_edge_medium） |
| 命名格式 | `episode_{task}_d{dist}_v{speed}_{bg}_{noise}_seed{N}` |

#### run002: 悬停距离梯度数据集

| 维度 | 值 |
|------|-----|
| Episode 总数 | 30 |
| 任务类型 | hover |
| 距离 | 50m, 100m, 150m, 200m, 250m, 300m, 350m, 400m, 450m, 500m |
| 每距离背景 | 3 种（empty_clean, flat_ground_mild, forest_edge_medium） |

### 4.6 13 种飞行任务类型

| # | 任务类型 | 最大速度 | 最大加速度 | 航点 (NED) | 描述 |
|---|----------|----------|------------|------------|------|
| 1 | hover | 5 m/s | 2 m/s² | [0,0,-40]→[0,0,-40] | 定点悬停 |
| 2 | approaching | 10 m/s | 3 m/s² | [80,0,-40]→[0,0,-40] | 向观察者接近 |
| 3 | receding | 10 m/s | 3 m/s² | [0,0,-40]→[80,0,-40] | 远离观察者 |
| 4 | linear_left | 10 m/s | 3 m/s² | [0,28,-40]→[0,-28,-40] | 向左线性飞行 |
| 5 | linear_right | 10 m/s | 3 m/s² | [0,-28,-40]→[0,28,-40] | 向右线性飞行 |
| 6 | diagonal | 10 m/s | 3 m/s² | [-15,-24,-40]→[30,24,-40] | 对角线穿越 |
| 7 | fast_crossing | 15 m/s | 5 m/s² | [12,-34,-40]→[12,34,-40] | 高速横向穿越（68m跨度） |
| 8 | accelerating | 12 m/s | 4 m/s² | [0,-30,-40]→[0,30,-40] | 加速飞行 |
| 9 | decelerating | 12 m/s | 4 m/s² | [0,-30,-40]→[0,30,-40] | 减速飞行 |
| 10 | sharp_turn | 10 m/s | 4 m/s² | [0,-28,-40]→[0,28,-40]→[44,28,-40] | 90度急转弯 |
| 11 | wide_turn | 10 m/s | 3 m/s² | [-2,-18,-40]→[42,18,-40] | 大弧度缓弯 |
| 12 | s_curve | 10 m/s | 4 m/s² | [-10,0,-40]→[15,24,-40]→[40,0,-40] | S形正弦曲线 |
| 13 | zigzag | 12 m/s | 5 m/s² | 5个交替航点 | 快速锯齿形 |

---

## 5. 分析引擎详解

### 5.1 数据流图

```
输入 Episode 目录
  │
  ├── episode.yaml ──────────+──> Stage 8 (元数据)
  ├── scenario.json          │
  ├── camera_info.json       │
  ├── frames.csv ───────────+──> Stage 1 (完整性) ──> integrity_report.json
  │                    │         Stage 2 (同步)   ──> sync_*.png, sync_frame_audit.csv
  │                    │         Stage 3 (阶段)   ──> phase_*.png, visibility_phase_audit.csv
  │                    │         Stage 7 (窗口)   ──> window_audit.csv, valid/invalid_windows.csv
  │                    │         Stage 8 (汇总)   ──> analysis_summary.json
  ├── truth.csv ────────────+──> Stage 1, 2, 3, 4
  ├── ideal_tracks.csv ─────+──> Stage 1, 5, 6, 7
  ├── measured_tracks.csv ──+──> Stage 1, 6, 7
  ├── events.jsonl ─────────+──> Stage 2 (事件解析, t_offset)
  └── rgb.mp4 ──────────────+──> Stage 1 (帧数验证)
```

### 5.2 分析指标详解

#### 时间同步指标

| 指标 | 说明 | 单位 |
|------|------|------|
| mean_ms | 平均同步误差 | ms |
| median_ms | 中位同步误差 | ms |
| p90_ms | 90分位同步误差 | ms |
| p95_ms | 95分位同步误差 | ms |
| p99_ms | 99分位同步误差 | ms |
| max_ms | 最大同步误差 | ms |
| invalid_ratio | 无效同步帧比例 | % |

#### 可见性指标

| 指标 | 说明 |
|------|------|
| visible_frames | 可见帧数 |
| visibility_ratio | 可见帧比例 |
| max_continuous_visible | 最大连续可见帧数 |
| max_continuous_invisible | 最大连续不可见帧数 |

#### 悬停质量指标

| 指标 | 说明 | 单位 |
|------|------|------|
| horizontal_rms_drift | 水平RMS漂移 | m |
| p95_radius | P95漂移半径 | m |
| max_drift | 最大漂移距离 | m |
| height_std | 高度标准差 | m |
| mean_horizontal_speed | 平均水平速度 | m/s |
| net_displacement | 净位移 | m |

#### 测量噪声指标

| 指标 | 说明 | 单位 |
|------|------|------|
| center_error_mean | 中心误差均值 | px |
| center_error_p95 | 中心误差P95 | px |
| center_error_max | 中心误差最大值 | px |
| missing_rate | 缺失检测率 | % |
| id_switches | ID切换次数 | count |
| system_bias_x/y | 系统偏差 | px |

---

## 6. 边缘端部署模块

### 6.1 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| `main.py` | 1173 | 核心应用：摄像头采集、运动检测、ROI提取、轨迹跟踪、推理调度、显示、网络 |
| `yololib.py` | 237 | RKNN模型封装：模型加载、输入预处理、多分支YOLO输出解码、NMS后处理 |
| `comms.py` | 67 | 网络通信：UDP视频流和JSON数据遥测 |

### 6.2 硬件平台

| 硬件 | 接口 | 用途 |
|------|------|------|
| **RK3588 NPU (3核)** | `rknnlite.api.RKNNLite` | YOLOv5 推理（6 TOPS） |
| **USB摄像头 (5路)** | V4L2 (`/dev/videoN`) | 2560x1440 MJPG @15fps |
| **云台** | UDP 192.168.0.100:8888 | 目标坐标 + 距离 JSON |
| **地面站** | UDP 192.168.0.200:9999-10003 | JPEG 视频流 |

### 6.3 线程架构

```
主线程 (显示)
  │
  ├── inference_worker 线程 (daemon)
  │     └── YoloRKNN 实例 (3核 NPU)
  │
  ├── capture_job[0] 线程 (daemon) ── 摄像头0
  ├── capture_job[1] 线程 (daemon) ── 摄像头1
  ├── capture_job[2] 线程 (daemon) ── 摄像头2
  ├── capture_job[3] 线程 (daemon) ── 摄像头3
  ├── capture_job[4] 线程 (daemon) ── 摄像头4
  │
  └── timer_job 线程 (daemon) ── 初始化定时器
```

**已知限制**: Python GIL 阻止 7 个线程在 RK3588 多核 CPU 上实现真正的并行。

### 6.4 运动引导级联检测管线

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
  |-- 评分排序 ROI (每帧最多4-6个)
  |-- 轨迹种子搜索 ROI (最多3个)
  |-- 创建 640x640 融合画布 (灰度+运动掩码)
  │
  v (inf_queues)
[inference_worker 线程] ── 单 YoloRKNN 实例
  |-- Resize 640x640 RGB
  |-- rknn.inference() 使用全部3核NPU
  |-- 解码YOLO输出 (6种布局变体)
  |-- 置信度过滤 + NMS
  │
  v (res_queues)
[capture_job 线程] ── 结果处理
  |-- 检测坐标映射回全帧
  |-- 喂入 TrajectoryFilter.update()
  |     |-- 多特征匹配 (距离+YOLO+模板+运动)
  |     |-- 速度预测, 动态门控
  |     |-- 轨迹确认 (YOLO直接/标准 双路径)
  |-- 对确认目标: 估算距离 (针孔模型)
  |-- 通过UDP JSON发送至云台
  |-- 绘制叠加 → 通过UDP JPEG发送视频
```

### 6.5 关键常量配置

#### 摄像头与帧

| 常量 | 值 | 说明 |
|------|-----|------|
| `N_CAM` | 5 | 摄像头数量 |
| `CAPTURE_W, CAPTURE_H` | 2560, 1440 | 原始采集分辨率 |
| `DIFF_W, DIFF_H` | 1920, 1080 | 帧差分分辨率 |
| `CROP_SIZE` | 640 | YOLO输入裁剪尺寸 |

#### 跟踪器

| 常量 | 值 | 说明 |
|------|-----|------|
| `TRACKER_MIN_HITS` | 4 | 确认轨迹最小命中数 |
| `TRACKER_MAX_DIST` | 100 | 基础最大匹配距离（像素） |
| `TRACKER_MAX_GATE` | 320 | 绝对最大门控距离 |
| `TRACKER_RECENT_WINDOW` | 18 | 命中历史滑动窗口 |

#### 距离估算

| 常量 | 值 | 说明 |
|------|-----|------|
| `ROUGH_TARGET_WIDTH_M` | 0.5 | 假设物理目标宽度（米） |
| `CAM_H_FOV` | 17.5 | 摄像头水平视场角（度） |
| `ROUGH_RANGE_MIN_M` | 20 | 最小估算距离 |
| `ROUGH_RANGE_MAX_M` | 2000 | 最大估算距离 |

### 6.6 TrajectoryFilter 多特征跟踪器

**匹配评分公式**:

```
match_score = 0.42 × distance_score
            + 0.23 × yolo_confidence
            + 0.17 × template_score
            + 0.18 × motion_score
```

**动态门控方程**:

```
gate = max_dist + 0.35 × speed × dt + 0.75 × physical_motion_px + 12.0 × misses
gate = clamp(gate, max_dist, max_gate)
```

**双路径确认机制**:

| 路径 | 条件 | 适用场景 |
|------|------|----------|
| YOLO直接确认 | yolo_hits≥3, recent_hits≥3, misses=0, score≥0.40 | 高置信度连续检测 |
| 标准确认 | hits≥min_hits, recent_hits≥min_recent, misses≤2, score≥0.50, traj_score≥0.55 | 轨迹平滑后的稳定目标 |

### 6.7 网络协议

#### 视频流（UDP）

```
字节 0:     0x01 (MSG_VIDEO)
字节 1:     cam_id (0-4)
字节 2-9:   board_id (UTF-8, 空字节填充至8字节)
字节 10+:   JPEG 压缩帧数据
```

#### 数据遥测（UDP JSON）

```json
{
  "board": "BOARD_2",
  "cam": 0,
  "type": "data",
  "objs": [[x1, y1, x2, y2, rough_range_m], ...]
}
```

### 6.8 性能特征

| 指标 | 值 |
|------|-----|
| 采集分辨率 | 2560x1440 (3.68 MP) |
| 处理分辨率 | 1920x1080 (2.07 MP) |
| YOLO输入 | 640x640 |
| 帧跳过 | 每3帧处理1帧 |
| 摄像头数量 | 5路同时 |
| 每帧最大ROI | 4-6 运动 + 3 轨迹种子 = 最多9个 |
| 推理模型 | yolov5s.rknn (YOLOv5-small, INT8) |
| NPU核心 | 3核 (6 TOPS) |
| 线程架构 | 7线程 (1推理 + 5采集 + 1定时) + 1主线程显示 |
| 背景学习 | 60帧 (~4秒) |
| 视频输出 | 640x360 JPEG quality=50, 每3帧 |
| 目标距离 | 20-2000米, 10米精度 |

---

## 7. GRU 轨迹预测模型

### 7.1 模型定位

GRU (Gated Recurrent Unit) 是 **YOLOv5 + ByteTrack + GRU** 三级管线中的第三阶段。它不读取图像，而是处理由检测器和跟踪器形成的 2D 轨迹序列。

**核心任务**:
1. 分类连续轨迹是真实无人机还是背景噪声
2. 从历史运动趋势预测未来 5 帧 2D 位置
3. 为目标过滤、轨迹维护和云台前馈控制提供数据

### 7.2 网络架构 (`UAVTrajectoryNet`)

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

### 7.3 输入格式

**Shape**: `[Batch, 20, 8]` — 20帧历史，每帧8个特征

| 特征 | 说明 |
|------|------|
| `x_norm`, `y_norm` | 归一化中心位置 (x/image_width, y/image_height) |
| `relative_x`, `relative_y` | 相对首帧的累积位移 |
| `velocity_x`, `velocity_y` | 逐帧速度 |
| `acceleration_x`, `acceleration_y` | 逐帧加速度 |

**预处理**: Savitzky-Golay 平滑（窗口=5, 多项式阶=2）用于抑制检测抖动

### 7.4 输出格式

| 输出 | Shape | 说明 |
|------|-------|------|
| 分类 | `[B, 1]` | logit → sigmoid → UAV 概率 (0.0~1.0) |
| 预测 | `[B, 5, 2]` | 未来5帧的相对偏移量（相对最后一帧历史位置） |

### 7.5 训练细节

| 项目 | 说明 |
|------|------|
| 损失函数 | `cls_weight × BCEWithLogitsLoss + reg_weight × SmoothL1Loss` |
| 回归损失 | 仅在正样本 (label=1) 上计算 |
| 标签 | 1=真实UAV, 0=非UAV, -1=未确认（不参与训练） |
| 梯度裁剪 | `max_norm=1.0` |
| 训练/测试划分 | 按 episode_id 划分（绝不随机打乱重叠窗口） |

### 7.6 推理逻辑

- 每个 Track ID 维护轨迹缓冲区，满20帧时触发推理
- **滞后状态机**: TENTATIVE → CONFIRMED (3次 ≥0.75) → SUPPRESSED (5次 ≤0.25) → LOST
- 部署方式: CPU (`torch.load(..., map_location="cpu")`)

---

## 8. 距离估算系统

### 8.1 摄像头参数

| 参数 | 值 |
|------|-----|
| 分辨率 | 1280x720 (debug模式) |
| 焦距 | fx=fy=4282.34 px |
| 水平视场角 | 17.0度 |
| 目标 | 3DR Iris 四旋翼 |
| 物理宽度 | 0.576m |
| 物理高度 | 0.205m |

### 8.2 尺度标定表

| 距离 | 像素宽度 | 像素高度 | 像素面积 | 图像覆盖率 |
|------|----------|----------|----------|------------|
| 50m | 49.39 px | 17.52 px | 865.2 px² | 0.094% |
| 100m | 24.66 px | 8.78 px | 216.5 px² | 0.024% |
| 300m | 8.22 px | 2.93 px | 24.1 px² | 0.003% |
| 500m | 4.91 px | 1.80 px | 8.9 px² | 0.001% |

### 8.3 距离估算公式

**像素面积法**:

```
d(Area) = 1471 / sqrt(Area)    [米]
```

**单轴法**:

```
W_pixel = K_w / d    (K_w ≈ 2470)
H_pixel = K_h / d    (K_h ≈ 876)
```

### 8.4 三种估算算法

| 算法 | 适用场景 | 方法 |
|------|----------|------|
| 像素面积距离估算 | YOLO后处理 | `1471 / sqrt(bbox_area)` |
| 卡尔曼滤波时序平滑 | 抑制单帧抖动 | 1D卡尔曼 (状态=[d, v_d], dt=0.0667s) |
| 级联模板匹配 | 距离 >400m | 32x32 ROI提取 + 5x2微模板互相关 |

---

## 9. 质量分级体系

### 9.1 四级评定标准

| 等级 | 条件 | 说明 |
|------|------|------|
| **A级** | HOVER阶段>200帧, valid_ratio>80%, 前15帧平均sync<1ms, 无异常sync | 优质训练数据 |
| **B级** | HOVER阶段>200帧, valid_ratio>50%, 前15帧平均sync<2ms | 良好训练数据 |
| **C级** | 有效窗口>50个 | 可清洗后使用 |
| **D级** | 总帧数=0, 或 sync无效比例>10%, 或有效窗口=0 | 不可用于训练 |

### 9.2 报告输出

#### analysis_report.md（61行）

1. **数据完整性审计** — CSV行对齐, NaN/Inf检查, 视频帧一致性
2. **时间同步** — 首帧延迟, 稳定期 mean/median/P95/P99/max 同步误差
3. **飞行阶段与可见性** — 可见帧数/比例, 可见/不可见区间
4. **世界运动与悬停质量** — 参考中心, RMS漂移, P95半径, 最大漂移, 高度标准差, 平均速度
5. **2D测量噪声** — 系统偏差, 中心误差 mean/P95/max, 缺失率, ID切换, 置信度
6. **20+5训练窗口审计** — 有效/无效窗口计数及失效类别分解

#### recommendations.md（30行）

1. **最终等级** — A/B/C/D评级及理由
2. **核心优势** — 检测精度, 悬停稳定性指标
3. **主要缺陷** — 无效窗口根因
4. **改进建议** — 操作修复（缩短ARM阶段, 优化初始摄像头朝向）

---

## 10. 技术栈与依赖

### 10.1 Python 依赖

| 类别 | 库 | 用途 |
|------|-----|------|
| **标准库** | os, csv, json, argparse, threading, queue, time, subprocess, math, collections | 基础操作 |
| **数值计算** | numpy | 数组运算, 矩阵操作 |
| **数据分析** | pandas | DataFrame, CSV处理 |
| **可视化** | matplotlib | 图表生成 (Agg后端) |
| **计算机视觉** | opencv-python (cv2) | 图像处理, 视频I/O, NMS |
| **配置解析** | PyYAML | YAML文件读取 |
| **RKNN推理** | rknnlite.api.RKNNLite | Rockchip NPU推理 |

### 10.2 硬件要求

| 组件 | 规格 |
|------|------|
| **边缘端** | RK3588 SoC (3核NPU, 6 TOPS) |
| **摄像头** | 5路 USB 摄像头, 2560x1440 MJPG @15fps |
| **存储** | 足够存储69个Episode (每个约数百MB) |
| **网络** | UDP通信 (云台192.168.0.100:8888, 地面站192.168.0.200:9999) |

### 10.3 开发环境

| 组件 | 规格 |
|------|------|
| **Python** | Anaconda (D:\anaconda3\python.exe) |
| **操作系统** | Windows (分析端) / Linux (边缘端RK3588) |
| **仿真** | PX4 SITL + Gazebo |

---

## 11. 代码质量分析

### 11.1 代码统计

| 文件 | 行数 | 函数/类数 | 复杂度 |
|------|------|-----------|--------|
| analyze_episode.py | 953 | 6函数 + 8阶段 | 高 |
| batch_analyze.py | 318 | 1函数 | 中 |
| main.py | 1173 | 25+函数 + 1类 | 极高 |
| yololib.py | 237 | 1类 (10+方法) | 高 |
| comms.py | 67 | 2类 | 低 |
| **总计** | **2748** | | |

### 11.2 已知问题

| 问题 | 文件 | 行号 | 严重度 | 说明 |
|------|------|------|--------|------|
| 死代码 | analyze_episode.py | 55 | 低 | `safe_float()` 定义但从未调用 |
| 未使用导入 | analyze_episode.py | 2 | 低 | `csv` 导入但未使用 |
| 裸except | analyze_episode.py | 58, 223 | 中 | 静默吞掉所有异常 |
| 硬编码Python路径 | batch_analyze.py | 10 | 高 | `D:\anaconda3\python.exe` 不可移植 |
| 硬编码时间戳 | analyze_episode.py | 247 | 中 | `1781083342.188` 仅适用于特定录制 |
| 无requirements.txt | 项目根目录 | - | 中 | 缺少依赖声明 |
| 无setup.py | 项目根目录 | - | 低 | 缺少包管理 |
| GIL限制 | main.py | 全局 | 高 | Python GIL阻止多核并行 |

### 11.3 代码亮点

1. **完善的七阶段分析管线**: 从数据完整性到训练窗口有效性，覆盖全面
2. **自适应路径解析**: 输出路径自动从输入路径推导，保持目录结构一致
3. **运动引导级联检测**: 多级滤波器有效减少误检
4. **双路径轨迹确认**: YOLO直接确认 + 标准轨迹确认，平衡速度和鲁棒性
5. **高可配置性**: 所有常量集中在模块顶部，便于按板卡调优

---

## 12. 部署与运维指南

### 12.1 分析端部署

```bash
# 单Episode分析
python analyze_episode.py -i "F:\UAV Trajectory System\export\input\run001\hover\episode_hover_d300_vhover_empty_clean_seed0001"

# 批量分析（所有run）
python batch_analyze.py

# 指定run分析
python batch_analyze.py -r run001
```

### 12.2 边缘端部署

```bash
# 确保RKNN模型文件存在
ls yolov5_model/../model/yolov5s.rknn

# 启动检测系统
cd yolov5_model
python main.py
```

### 12.3 网络配置

| 端点 | IP | 端口 | 协议 |
|------|-----|------|------|
| 云台控制 | 192.168.0.100 | 8888 | UDP JSON |
| 视频流 | 192.168.0.200 | 9999-10003 | UDP JPEG |

### 12.4 性能调优

**UDP缓冲区调优** (Linux):

```bash
sudo sysctl -w net.core.rmem_max=8388608
sudo sysctl -w net.core.rmem_default=2097152
```

---

## 13. 改进建议与路线图

### 13.1 短期改进

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P0 | 添加 requirements.txt | 声明所有Python依赖 |
| P0 | 移除硬编码路径 | 使用环境变量或配置文件 |
| P1 | 清理死代码 | 删除 `safe_float()`, 移除未使用 `csv` 导入 |
| P1 | 改进异常处理 | 替换裸 `except` 为具体异常类型 |
| P2 | 添加单元测试 | 为关键函数编写测试用例 |

### 13.2 中期改进

| 优先级 | 改进项 | 说明 |
|--------|--------|------|
| P1 | RKNN多进程重构 | 使用 multiprocessing 替代 threading，绕过GIL |
| P1 | NPU核心分配 | Core 0+1 给YOLOv5, Core 2 给轨迹模型 |
| P2 | 配置外部化 | 将硬编码常量提取到 YAML/TOML 配置文件 |
| P2 | 日志系统 | 添加结构化日志替换 print 输出 |

### 13.3 长期路线图

| 阶段 | 任务 | 预期效果 |
|------|------|----------|
| Phase 1: 模型转换 | ONNX → RKNN (rknn-toolkit2) | 精度损失 < 1.5% |
| Phase 2: Python重构 | multiprocessing pool + NPU核心分离 | 5路摄像头 10-15 FPS |
| Phase 3: C++重写 | MPP硬件解码 + RGA硬件预处理 + 零拷贝推理 | >25 FPS, CPU < 30% |

### 13.4 C++ 重写路线图

| 组件 | 当前 (Python) | 目标 (C++) | 预期收益 |
|------|---------------|------------|----------|
| 视频解码 | V4L2 + OpenCV | MPP 硬件解码 | 5-10x 加速 |
| 图像预处理 | OpenCV resize/cvtColor | RGA 硬件加速 | <1ms |
| 推理 | rknnlite Python API | DMA-BUF 零拷贝 | 减少内存拷贝 |
| 整体 | 7线程 Python | 多进程 C++ | >25 FPS |

---

## 附录 A: 文件索引

| 文件 | 路径 | 行数 | 说明 |
|------|------|------|------|
| analyze_episode.py | `./analyze_episode.py` | 953 | 单Episode分析引擎 |
| batch_analyze.py | `./batch_analyze.py` | 318 | 批量分析调度器 |
| main.py | `./yolov5_model/main.py` | 1173 | YOLOv5多摄像头检测 |
| yololib.py | `./yolov5_model/yololib.py` | 237 | RKNN推理封装 |
| comms.py | `./yolov5_model/comms.py` | 67 | UDP通信 |
| GRU说明 | `./document/GRU_无人机时序轨迹网络详细说明.md` | 1662 | GRU模型文档 |
| 技能定义 | `./agent/skill/uav_data_analysis_skill.md` | 138 | AI Agent技能 |
| 距离规范 | `./distance/distance_scale_specification.md` | 184 | 距离标定 |
| 系统审计 | `./yolov5_model/uav_yolo_rknn_report.md` | 239 | YOLOv5审计报告 |

---

> **文档结束** — 此文档由 MiMo Code 自动生成，基于对项目全部源码、文档和数据的深度分析。
