# 无人机测距系统（run017）数据构成与模型架构分析报告

本报告旨在对 `F:\UAV Trajectory System\distance data` 目录下的文件构成、物理估算模型、残差 MLP 架构、数据类型及整体处理逻辑进行深入剖析，并针对物理迁移与下一阶段的模型训练提供系统性评估与改进建议。

---

## 一、 文件夹目录结构与文件构成

整个训练包采用清晰的模块化设计，将原始仿真数据、预处理特征数据集、模型参数、评测指标与工具链完全解耦：

```text
distance data/
├── README_WINDOWS_TRAINING.md      # Windows 训练包说明文件
├── TRAINING_PLAN_WINDOWS.md        # 详细的训练特征契约与评估计划
├── requirements_windows.txt        # 最小运行依赖配置文件（仅 NumPy, Pandas 等）
├── export_manifest.json            # 导出运行数据 run017 的全局元数据清单
├── calibration/
│   └── camera_profile.json         # 相机标定内参及目标物体的物理尺寸
├── input/
│   ├── manifests/                  # 包含点级和会话级的全局索引清单
│   │   ├── point_manifest_windows.csv
│   │   └── session_manifest_windows.csv
│   ├── distance_points/run017/     # 按场景和距离分块存储的帧级特征/真实值 CSV 文件夹
│   └── raw_sessions/run017/        # 原始 PX4 SITL + Gazebo ROS 图像主题会话记录
├── dataset/
│   └── run017_windows_v1/          # 生成的 frame_dataset.csv 及 train/val/test 划分
├── models/
│   └── run017_windows_v1/          # 训练好的残差模型参数（NumPy .npz 格式）与元数据
├── reports/
│   └── run017_windows_v1/          # 物理基线与神经网络模型的评测指标 JSON
├── src/uav_distance_pipeline/      # 核心管道逻辑源码
└── tools/                          # 阶段执行工具（生成数据集、评估基线、训练残差模型）
```

---

## 二、 系统架构与计算模型

测距系统采用 **“几何物理模型 + 机器学习残差修正”** 的混合测距架构。

### 1. 相机几何与物理基线
系统根据针孔相机投影模型建立测距基线，参数定义在 [camera_profile.json](file:///F:/UAV%20Trajectory%20System/distance%20data/calibration/camera_profile.json) 中：
- **图像分辨率**：$2560 \times 1440$ 像素
- **焦距参数**：$f_x = f_y = 11494.25$
- **光心坐标**：$c_x = 1280.0$，$c_y = 720.0$
- **目标无人机物理尺寸**：宽度 $W_{uav} = 0.30$ 米，高度 $H_{uav} = 0.15$ 米

物理模型分别计算出三个独立的测距估计值，并求取平均值作为物理融合距离 `fused_range_m`：
- **基于宽度估算**：
  $$D_w = \frac{f_x \cdot W_{uav}}{width_{pixel}}$$
- **基于高度估算**：
  $$D_h = \frac{f_y \cdot H_{uav}}{height_{pixel}}$$
- **基于面积估算**：
  $$D_{area} = \sqrt{\frac{f_x \cdot f_y \cdot W_{uav} \cdot H_{uav}}{width_{pixel} \cdot height_{pixel}}}$$
- **均值融合**：
  $$D_{fused} = \frac{D_w + D_h + D_{area}}{3}$$

### 2. 神经网络残差修正架构
物理模型能处理大部分的尺度缩放，但无法建模风偏、振动、像素量化误差及复杂的场景噪声。MLP（多层感知机）负责拟合这些非线性残差：
- **目标残差（标签）**：
  $$R_{residual} = R_{true} - D_{fused}$$
- **网络输入**：包括包围盒几何（长、宽、面积、中心点坐标、检测置信度）、物理估算值、nominal 距离区间分箱以及当前场景环境一热码（3种场景）。
- **网络结构**：单隐藏层 MLP，输入维度 16，隐藏层 32（采用 ReLU 激活），输出层维度 1。
- **最终测距预测值**：
  $$R_{predicted} = D_{fused} + \hat{R}_{residual}$$

---

## 三、 数据类型与特征契约

模型输入特征在 [feature_schema.json](file:///F:/UAV%20Trajectory%20System/distance%20data/dataset/run017_windows_v1/feature_schema.json) 中约束，特征及数据类型细分如下：

| 数据分组 | 字段名称 | 数据类型 | 物理含义 |
| :--- | :--- | :--- | :--- |
| **检测盒几何** | `bbox_width_px`, `bbox_height_px` | Float | 检测出的包围盒宽、高（像素） |
| | `bbox_area_px` | Float | 检测包围盒的面积（像素） |
| | `bbox_center_u`, `bbox_center_v` | Float | 像素坐标系中包围盒的中心坐标 |
| | `detector_confidence` | Float | 检测置信度（仿真默认为 0.98~1.0） |
| **物理粗估** | `physics_width_range_m` | Float | 基于宽度计算的距离（米） |
| | `physics_height_range_m` | Float | 基于高度计算的距离（米） |
| | `physics_area_range_m` | Float | 基于面积计算的距离（米） |
| | `fused_range_m` | Float | 三者融合后的测距预测值（米） |
| **分箱与元数据** | `nominal_distance_m` | Float | 飞行的标称预设距离 |
| | `distance_bin_start_m`, `distance_bin_end_m` | Float | 距离范围区间上下界（分箱） |
| **场景一热码** | `scene_empty_clean_no_wind` | Binary (0/1) | 无风且干净背景场景 |
| | `scene_flat_ground_light_wind` | Binary (0/1) | 平地轻风场景 |
| | `scene_forest_edge_moderate_wind`| Binary (0/1) | 林缘中风场景 |
| **模型标签** | `true_range_m` | Float | 物理真实距离 $R_{true}$（用于验证） |
| | `residual_range_m` | Float | 残差真实值 $R_{true} - D_{fused}$（训练目标） |

---

## 四、 下一阶段模型训练的评估与改进建议

### 1. 当前优势分析
- **高效的无框架部署能力**：`train_distance_model.py` 内部手动实现了 MLP 的前向传播和反向传播（纯 NumPy 编写）。模型训练完毕后，以简易的 `.npz` 格式存储权重。这使得测距系统能够脱离 PyTorch/TensorFlow 依赖，极易在 C++、PX4 伴机电脑或 RK3588 开发板上无缝集成。
- **物理先验保障泛化下限**：通过回归残差而非绝对数值，模型在特征未覆盖的超远距离区域（如 >500m）依然能够提供物理模型的预测基线，这大大降低了深度网络在长尾分布中直接退化或输出异常预测值的风险。

### 2. 物理迁移与部署风险识别（重点）

#### A. 像素量化误差与远距离“距离抖动”
- **机制风险**：仿真生成的包围盒极为精确，但在真实运行中检测器边界会产生 1~3 px 的随机抖动。在超远距离（如 400m-500m）下，无人机成像宽度仅约 8 px。**1 个像素的宽度扰动（8 px 变为 9 px）将导致测距误差瞬间飙升至近 40 米**。
- **对策建议**：
  1. **数据增强**：在下一阶段的训练数据集生成中，对输入特征 `bbox_width_px`、`bbox_height_px` 叠加随机的高斯噪声和量化截断（Quantization Rounding）。
  2. **引入时序状态机**：单帧测距不可避免地会受噪声干扰，下一阶段模型应结合历史多帧输出，通过轻量级滤波（如 Kalman Filter）或短时序窗口（如 5-10 帧的滑动均值）来消除跳变。

#### B. 无人机姿态倾斜造成的几何投影收缩
- **机制风险**：当无人机发生横滚（Roll）或俯仰（Pitch）机动时，在相机视角中，其长宽比和二维边界投影会由于视角投影（Perspective Projection）发生收缩。物理模型将收缩后的检测框误认为是目标离得更远，导致测距估计偏高。
- **对策建议**：
  1. **引入特征自适应能力**：在训练特征中，加入包围盒宽高的比值（`width / height`）来指示形变，或是直接将无人机回传的实时姿态角（Pitch, Roll）合并到特征向量中，作为物理修正的特征输入。
  2. **训练样本多样化**：在下一阶段的数据收集中，必须补充大倾角机动飞行场景的数据，防止模型退化为纯静态测距。

#### C. 网络表征能力的扩展
- **机制风险**：当前 MLP 采用极简的 1 层隐藏层（32 节点）。仿真环境的场景比较干净单一，但当真实数据中加入光照、镜头畸变、多类气象条件时，当前模型的容量可能会遭遇表征瓶颈（欠拟合）。
- **对策建议**：
  1. **模型适度加宽**：将隐藏层结构调整为两层（如 16 $\to$ 64 $\to$ 32 $\to$ 1），并在层间加入 Dropout（比例建议为 0.1~0.2），提升模型对传感器突变噪声的容忍度。
  2. **强化标定校准**：如果相机焦距或畸变参数有微调，应通过修改 `camera_profile.json` 自动更新物理测距基线，而无需重新训练整个残差神经网络，这也是混合测距系统的设计优势所在。
