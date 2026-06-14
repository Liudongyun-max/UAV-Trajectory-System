# UAV 轨迹预测与可视化检测系统 — 完整部署指南

> **版本**: v1.0 | **更新日期**: 2026-06-13

---

## 目录

1. [系统概述](#1-系统概述)
2. [环境配置一键脚本](#2-环境配置一键脚本)
3. [手动配置指南](#3-手动配置指南)
4. [使用方法](#4-使用方法)
5. [文件结构说明](#5-文件结构说明)
6. [配置参数详解](#6-配置参数详解)
7. [输出说明](#7-输出说明)
8. [常见问题](#8-常见问题)

---

## 1. 系统概述

### 1.1 功能简介

本系统是一个基于 GRU 时序轨迹网络的无人机目标检测与预测可视化工具，具备：

- **实时视频推理**: 逐帧处理视频，实时显示预测结果
- **轨迹分类**: 判断目标是真实无人机还是背景噪声
- **未来轨迹预测**: 基于20帧历史预测未来5帧位置
- **双模式运行**: 支持 CSV 数据驱动模式和纯视频驱动模式
- **可视化输出**: 历史轨迹、预测轨迹、真实轨迹叠加显示

### 1.2 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| GUI框架 | PyQt5 |
| 视频处理 | OpenCV |
| 数据处理 | pandas, numpy |
| 信号平滑 | scipy (Savitzky-Golay) |
| 模型架构 | GRU (Gated Recurrent Unit) |

---

## 2. 环境配置一键脚本

### 2.1 Windows 一键配置（推荐）

将以下内容保存为 `setup_env.bat`，放在 `gru detect` 目录下，双击运行即可：

```bat
@echo off
chcp 65001 >nul
echo ========================================
echo   UAV 轨迹预测系统 - 环境配置脚本
echo ========================================
echo.

REM 检查 conda 是否可用
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 未检测到 conda，请先安装 Anaconda 或 Miniconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/4] 创建 conda 环境 'uav_gru' (Python 3.10)...
conda create -n uav_gru python=3.10 -y
if %errorlevel% neq 0 (
    echo [ERROR] 创建环境失败
    pause
    exit /b 1
)

echo [2/4] 激活环境...
call conda activate uav_gru

echo [3/4] 安装依赖包...
pip install numpy pandas opencv-python torch scipy PyQt5 -q
if %errorlevel% neq 0 (
    echo [ERROR] 安装依赖失败
    pause
    exit /b 1
)

echo [4/4] 验证安装...
python -c "import torch; import cv2; import PyQt5; print('所有依赖安装成功!')"
if %errorlevel% neq 0 (
    echo [ERROR] 验证失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo   配置完成! 
echo   请使用以下命令启动:
echo   conda activate uav_gru
echo   python predict_gui.py
echo ========================================
pause
```

### 2.2 修改启动脚本

配置完成后，需要修改 `启动UAV预测窗口.vbs` 中的环境名称：

```vbs
Set shell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
cmd = "cmd /c set KMP_DUPLICATE_LIB_OK=TRUE && set OMP_NUM_THREADS=1 && cd /d """ & scriptDir & """ && conda run -n uav_gru pythonw """ & scriptDir & "\predict_gui.py"""
shell.Run cmd, 0, False
```

**注意**: 将 `yolo` 改为你实际创建的环境名称（如 `uav_gru`）。

---

## 3. 手动配置指南

### 3.1 前置要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.8+ | 推荐 3.10 |
| Anaconda/Miniconda | 最新 | 推荐使用 conda 管理环境 |
| NVIDIA GPU | 可选 | CUDA 加速推理（无GPU则使用CPU） |
| CUDA Toolkit | 11.7+ | 如果使用 NVIDIA GPU |

### 3.2 创建环境

```bash
# 创建新环境
conda create -n uav_gru python=3.10

# 激活环境
conda activate uav_gru
```

### 3.3 安装依赖

```bash
# 进入项目目录
cd "F:\UAV Trajectory System\gru detect"

# 安装依赖
pip install -r requirements.txt
```

### 3.4 依赖清单

```
numpy          # 数值计算
pandas         # 数据处理
opencv-python  # 视频处理
torch          # PyTorch深度学习框架
scipy          # 信号平滑 (Savitzky-Golay)
PyQt5          # 图形界面
```

### 3.5 验证安装

```bash
python -c "
import torch
import cv2
import PyQt5
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'OpenCV: {cv2.__version__}')
print('所有依赖验证通过!')
"
```

---

## 4. 使用方法

### 4.1 一键启动（推荐）

双击运行：
```
启动UAV预测窗口.vbs
```

### 4.2 命令行启动

```bash
# 激活环境
conda activate uav_gru

# 启动GUI
python predict_gui.py
```

### 4.3 操作步骤

1. **导入视频**: 点击"浏览..."选择待检测的视频文件（支持 .mp4, .avi, .mkv, .mov）

2. **选择模型**: 从下拉框选择预训练模型，或点击"自定义权重..."加载自己的模型

3. **配置CSV（可选）**: 
   - 勾选"启用测量轨迹 CSV"
   - 选择 `measured_tracks.csv` 文件
   - 系统会自动加载同目录下的 `ideal_tracks.csv` 作为真实轨迹

4. **设置参数**:
   - 开启 Savgol 历史轨迹平滑（推荐）
   - 保存预测视频到 outputs/
   - 选择显示内容（预测轨迹、历史轨迹、真实轨迹）
   - 选择计算设备（auto/cuda/cpu）

5. **启动检测**: 点击"启动实时检测与预测"

6. **监控进度**: 观察右侧视频预览和控制台日志

7. **停止与保存**: 点击"停止并保存"或直接关闭窗口

---

## 5. 文件结构说明

```
gru detect/
│
├── predict_gui.py              # 主程序 - PyQt5 GUI 入口
├── 启动UAV预测窗口.vbs         # Windows 一键启动脚本
├── requirements.txt            # Python 依赖清单
├── README.md                   # 原始说明文档
├── DEPLOYMENT_GUIDE.md         # 本文档 - 完整部署指南
│
├── src/                        # 核心源码目录
│   ├── model.py                # GRU 模型定义 (UAVTrajectoryNet)
│   ├── inference.py            # 推理会话管理 (InferenceSession)
│   ├── tracker.py              # 特征追踪器 (FeatureTracker)
│   └── paths.py                # 路径工具函数
│
├── models/                     # 预训练模型目录
│   ├── gru_baseline.pth        # GRU 基线模型
│   └── gru_enhanced.pth        # GRU 增强模型
│
└── outputs/                    # 输出目录（自动创建）
    └── {episode_name}/
        ├── {episode_name}_predicted.avi  # 预测视频
        └── prediction_metrics.csv        # ADE 指标
```

### 5.1 核心模块说明

#### model.py — 模型定义

```python
class UAVTrajectoryNet(nn.Module):
    """
    GRU 轨迹预测网络
    
    输入: [Batch, 20, 8] — 20帧历史，8维特征
    输出: 
      - logits: [Batch, 1] — 分类概率（真/伪无人机）
      - future_offsets: [Batch, 5, 2] — 未来5帧偏移
    """
```

**8维输入特征**:

| 特征 | 说明 |
|------|------|
| x_norm | 归一化 X 坐标 |
| y_norm | 归一化 Y 坐标 |
| relative_x | 相对首帧 X 位移 |
| relative_y | 相对首帧 Y 位移 |
| velocity_x | 帧间 X 速度 |
| velocity_y | 帧间 Y 速度 |
| acceleration_x | 帧间 X 加速度 |
| acceleration_y | 帧间 Y 加速度 |

#### inference.py — 推理会话

管理视频读取、模型推理、视频写入的完整生命周期：

- `_open()`: 初始化模型、视频、追踪器
- `step()`: 处理单帧
- `_render_frame()`: 渲染预测结果
- `close()`: 保存输出并清理资源

#### tracker.py — 特征追踪器

双模式目标追踪：

1. **CSV 数据驱动模式**: 使用 CSV 中的坐标，结合视觉偏置修正
2. **纯视频驱动模式**: 基于灰度阈值的目标检测与追踪

核心功能：
- 目标检测 (`detect_drone_centroid`)
- 历史缓冲区管理
- 偏置对齐修正
- 死锁检测与恢复

---

## 6. 配置参数详解

### 6.1 GUI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 启用测量轨迹 CSV | True | 是否使用 CSV 数据驱动模式 |
| 开启 Savgol 平滑 | True | 对历史轨迹进行 Savitzky-Golay 平滑 |
| 保存预测视频 | True | 是否将预测结果保存为视频 |
| 展示预测轨迹 | True | 显示蓝色预测轨迹线 |
| 展示历史轨迹 | True | 显示红色历史轨迹线 |
| 展示真实轨迹 | True | 显示绿色真实轨迹线（需 CSV） |
| 计算设备 | auto | auto/cuda/cpu |

### 6.2 模型参数

| 参数 | 基线模型 | 增强模型 |
|------|----------|----------|
| input_dim | 8 | 12 |
| hidden_dim | 32 | 64 |
| num_layers | 1 | 2 |
| future_steps | 5 | 5 |
| dropout | 0.1 | 0.2 |
| 参数量 | ~5K | ~25K |

### 6.3 追踪器参数

| 参数 | 值 | 说明 |
|------|-----|------|
| seq_len | 20 | 历史窗口长度 |
| smooth | True | 是否启用平滑 |
| normalize | True | 是否归一化 |
| max_offset_jump | 40.0 | 偏置跳跃门限（像素） |
| max_drone_move | 60.0 | 单帧最大位移门限 |

---

## 7. 输出说明

### 7.1 输出目录结构

```
outputs/
└── {episode_name}/
    ├── {episode_name}_predicted.avi   # 带预测叠加的视频
    └── prediction_metrics.csv         # 逐帧 ADE 指标
```

### 7.2 视频输出

- 格式: AVI (MJPG 编码)
- 分辨率: 与输入视频相同
- 内容: 原始帧 + 叠加的轨迹线和 HUD 信息

### 7.3 叠加显示说明

| 颜色 | 含义 |
|------|------|
| 红色折线 | 历史轨迹（过去20帧） |
| 蓝色折线/点 | GRU 预测轨迹（未来5帧） |
| 绿色折线/点 | 真实轨迹（GT，需 CSV） |
| 黄色矩形 | 当前目标检测框 |
| 左上角 HUD | 帧号、分类状态、ADE 值 |

### 7.4 prediction_metrics.csv

```csv
frame,ade_px,pred_points,gt_points
0,12.34,"640,360;645,358;650,355;...","641,359;646,357;651,354;..."
1,11.21,"645,358;650,355;655,352;...","646,357;651,354;656,351;..."
```

---

## 8. 常见问题

### Q1: 启动时报错 "No module named 'PyQt5'"

```bash
pip install PyQt5
```

### Q2: 启动时报错 "No module named 'torch'"

```bash
pip install torch
# 如果需要 CUDA 版本:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q3: 视频无法播放或黑屏

检查视频格式是否支持：
```python
import cv2
cap = cv2.VideoCapture("your_video.mp4")
print(f"是否打开: {cap.isOpened()}")
print(f"帧数: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
cap.release()
```

### Q4: 模型加载失败

确保模型文件完整：
```python
import torch
checkpoint = torch.load("models/gru_baseline.pth", map_location="cpu")
print(checkpoint.keys())
```

### Q5: 预测结果不准确

1. 确保使用正确的模型（基线 vs 增强）
2. 检查 CSV 文件是否与视频对应
3. 尝试开启/关闭 Savgol 平滑

### Q6: 运行速度慢

1. 使用 GPU 加速：
   - 安装 CUDA 版本 PyTorch
   - 在 GUI 中选择 `cuda` 设备

2. 降低视频分辨率

### Q7: 如何使用自己的训练模型

1. 训练模型并保存为 `.pth` 格式
2. 确保模型配置包含：
   ```python
   checkpoint = {
       "model_state_dict": model.state_dict(),
       "model_config": {
           "input_dim": 8,
           "hidden_dim": 32,
           "num_layers": 1,
           "future_steps": 5,
           "dropout": 0.1,
       }
   }
   ```
3. 将模型放入 `models/` 目录或通过 GUI 自定义选择

### Q8: 错误日志位置

GUI 异常日志保存在：
```
gru detect/predict_gui_error.log
```

---

## 9. 快速开始（5步）

```bash
# 1. 创建环境
conda create -n uav_gru python=3.10 -y
conda activate uav_gru

# 2. 安装依赖
cd "F:\UAV Trajectory System\gru detect"
pip install numpy pandas opencv-python torch scipy PyQt5

# 3. 启动 GUI
python predict_gui.py

# 4. 在 GUI 中选择视频和模型

# 5. 点击"启动实时检测与预测"
```

或直接双击 `启动UAV预测窗口.vbs`（需先修改其中的环境名称）。

---

## 10. 联系与支持

如有问题，请检查：
1. `DEPLOYMENT_GUIDE.md`（本文档）
2. `README.md`（原始说明）
3. `predict_gui_error.log`（错误日志）
