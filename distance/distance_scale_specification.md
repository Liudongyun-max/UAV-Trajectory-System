# -*- coding: utf-8 -*-
# 无人机视觉测距与尺度映射设计协议及工程落地指南

本文件定义了基于单目相机成像尺度的无人机物理距离反解协议，并为后续在 **RK3588** 板端进行算法嵌入和 C++ 移植提供详细的数学推导与工程实现蓝图。

---

## 一、 仿真系统核心光学与物理参数

在当前 Gazebo 数据采集系统中，相机与待测无人机的物理及光学基准参数如下：

1. **相机内参 (Camera Intrinsics)**：
   * 图像分辨率：$1280 \times 720$ 像素（宽度 $W = 1280$，高度 $H = 720$）。
   * 水平/垂直焦距（主焦距）：$f_x = f_y = 4282.34$ 像素。
   * 水平视场角 (HFOV)：$17.0^\circ$（窄角长焦镜头，用于远距离跟瞄）。
2. **待测无人机物理尺寸 (3DR Iris)**：
   * 物理外包络宽度 ($W_{phys}$)：**$0.576$ 米** ($57.6\text{ cm}$，含电机与旋翼外廓)。
   * 物理外包络高度 ($H_{phys}$)：**$0.205$ 米** ($20.5\text{ cm}$，包含高脚起落架)。

---

## 二、 尺度变化量化审计数据 (基于 run002 标定)

以下数据基于 `run002` 悬停数据集提取，展示了距离 $d$ 与可观测像素尺寸的对应关系：

| 距离 (d) | 平均像素宽度 ($W_{pixel}$) | 平均像素高度 ($H_{pixel}$) | 像素面积 ($Area_{pixel}$) | 图像像素占比 |
| :---: | :---: | :---: | :---: | :---: |
| **50 米** | 49.39 px | 17.52 px | 865.2 px^2 | 0.0939% |
| **100 米** | 24.66 px | 8.78 px | 216.5 px^2 | 0.0235% |
| **150 米** | 16.45 px | 5.84 px | 96.0 px^2 | 0.0104% |
| **200 米** | 12.33 px | 4.38 px | 54.0 px^2 | 0.0059% |
| **250 米** | 9.86 px | 3.51 px | 34.6 px^2 | 0.0038% |
| **300 米** | 8.22 px | 2.93 px | 24.1 px^2 | 0.0026% |
| **350 米** | 7.04 px | 2.52 px | 17.7 px^2 | 0.0019% |
| **400 米** | 6.17 px | 2.21 px | 13.6 px^2 | 0.0015% |
| **450 米** | 5.49 px | 1.96 px | 10.8 px^2 | 0.0012% |
| **500 米** | 4.91 px | 1.80 px | 8.9 px^2 | 0.0010% |

---

## 三、 数学模型与拟合公式

根据针孔投影原理，像素尺寸与距离成反比：
$$W_{pixel} = \frac{W_{phys} \cdot f_x}{d} = \frac{K_w}{d}$$
$$H_{pixel} = \frac{H_{phys} \cdot f_y}{d} = \frac{K_h}{d}$$

### 1. 单维度尺度常数
* 水平尺度常数：$K_w = W_{phys} \cdot f_x \approx \mathbf{2470} \text{ px}\cdot\text{m}$
* 垂直尺度常数：$K_h = H_{phys} \cdot f_y \approx \mathbf{876} \text{ px}\cdot\text{m}$

### 2. 双维度面积尺度常数
由于像素长宽在远距离下存在离散化舍入噪声，采用**检测框像素面积**进行拟合可极大增强鲁棒性：
$$Area_{pixel} = W_{pixel} \cdot H_{pixel} = \frac{K_w \cdot K_h}{d^2} \approx \frac{\mathbf{2,163,720}}{d^2}$$

解算出距离 $d$ 与像素面积的映射公式：
$$d(Area_{pixel}) = \frac{\mathbf{1471}}{\sqrt{Area_{pixel}}} \quad (\text{米})$$

---

## 四、 改造实施与工程落地指南

为解决目标远距离抖动、量化阶跃并提升在 **RK3588** 平台上的闭环控制精度，建议采取以下三项低成本补强措施。

### 1. 算法一：基于“像素面积反比例解算”的测距模块
在 YOLOv5 的推理后处理代码中（例如 `yololib.py` 的最后），添加基于面积的快速求解逻辑：

```python
def calculate_distance_by_area(x1, y1, x2, y2):
    """
    基于 Bounding Box 面积计算物理距离 (单位: 米)
    """
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    
    if area <= 0.0:
        return -1.0 # 无效距离
        
    # 基于 run002 标定的拟合系数 1471
    distance = 1471.0 / np.sqrt(area)
    return float(distance)
```

### 2. 算法二：卡尔曼滤波器（Kalman Filter）时序测距平滑
在主跟踪器（`TrajectoryFilter`）中，引入一维卡尔曼滤波器对估算的距离 $d$ 进行时序状态平滑，防止因单帧检测框边缘抖动产生的高频阶跃：

```python
class DistanceKalmanFilter:
    def __init__(self, init_d, q_noise=0.1, r_noise=1.5):
        # 状态量 x = [d, v_d] (距离, 距离变化速度)
        self.x = np.array([[init_d], [0.0]])
        # 状态转移矩阵 (dt = 0.0667s, 对应 15 FPS)
        self.dt = 0.0667
        self.A = np.array([[1.0, self.dt],
                           [0.0, 1.0]])
        # 观测矩阵 H (只观测距离 d)
        self.H = np.array([[1.0, 0.0]])
        # 协方差矩阵 P
        self.P = np.eye(2) * 1.0
        # 过程噪声协方差 Q
        self.Q = np.array([[q_noise, 0.0],
                           [0.0, q_noise * 0.1]])
        # 测量噪声协方差 R
        self.R = np.array([[r_noise]])

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0, 0]

    def update(self, z_d):
        # 观测残留
        y = z_d - np.dot(self.H, self.x)
        # S = H P H_T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # 卡尔曼增益 K
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # 更新状态
        self.x = self.x + np.dot(K, y)
        # 更新协方差
        self.P = np.dot((np.eye(2) - np.dot(K, self.H)), self.P)
        return self.x[0, 0]
```

### 3. 算法三：极远距离下的“传统模板匹配后处理微调”级联模块
当检测距离 $d > 400\text{m}$（面积 $Area_{pixel} < 14\text{ px}^2$）时，YOLOv5 检测边界极易受背景干扰跳变。此时采用级联机制：以 YOLO 检测中心为基准，提取 $32 \times 32$ 局部 ROI 图像，使用微型无人机梯度模板进行互相关对齐，修正定位偏差。

```python
class CascadeTemplateRefiner:
    def __init__(self):
        # 建立极远距离下的标准微型模板 (基于物理尺寸投影，通常 5x2 像素)
        # 这里可以使用一个简化的 H 样特征或小二乘模板
        self.base_template = np.array([
            [0, 255, 0, 255, 0],
            [255, 255, 255, 255, 255]
        ], dtype=np.uint8)

    def refine_center(self, img_bgr, yolo_cx, yolo_cy):
        """
        在 YOLO 中心点周围进行传统模板二次精细定位 (亚像素微调)
        """
        h_img, w_img = img_bgr.shape[:2]
        roi_size = 16 # 限制局部搜索半径在 16px 窗口内
        
        # 1. 裁剪局部 ROI
        x1 = max(0, int(yolo_cx - roi_size))
        y1 = max(0, int(yolo_cy - roi_size))
        x2 = min(w_img, int(yolo_cx + roi_size))
        y2 = min(h_img, int(yolo_cy + roi_size))
        
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return yolo_cx, yolo_cy
            
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 2. 局部模板匹配
        res = cv2.matchTemplate(roi_gray, self.base_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # 3. 若相关性得分达标，则修正中心点
        if max_val > 0.65:
            # 模板左上角偏移对应的中心偏移
            match_x = max_loc[0] + self.base_template.shape[1] / 2.0
            match_y = max_loc[1] + self.base_template.shape[0] / 2.0
            refined_cx = x1 + match_x
            refined_cy = y1 + match_y
            return refined_cx, refined_cy
            
        return yolo_cx, yolo_cy
```

---

## 五、 后期精细化关系拟合与校准路线

为使得真实复杂环境下的视觉测距模型达到工业级精度，后期需按照以下路线进行数据拟合与校准：

1. **多大气能见度场景数据注入**：
   在物理测试阶段，采集不同气象能见度（Clean / Mild / Foggy）下的无人机飞行动作，通过曲线拟合工具对不同光影条件下的面积系数 $K$ 进行动态环境温度/对比度加权校正。
2. **镜头畸变消除 (Distortion Undistortion)**：
   在进行距离换算前，必须使用 OpenCV 内参标定接口 `cv2.undistortPoints` 对检测出的 `projected_u` 和 `projected_v` 进行去畸变处理，特别是边缘视场（由于广角或镜头周边畸变会导致像素压缩，引发严重的测距漂移）。
3. **闭环融合嵌入**：
   最终解算出的时序平滑物理距离 $d$，应直接作为轨迹预测模型（LSTM / TCN）的输入维度之一，辅助时序网络判断无人机当前是处于“逼近（approaching）”、“远离（receding）”还是“稳态悬停（hover）”物理模式。
