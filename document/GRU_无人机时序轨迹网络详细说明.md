# GRU 时序轨迹网络原理、运行逻辑与无人机项目落地说明

## 1. 文档目的

本文面向当前“远距离点状无人机轨迹采集、判定与预测”项目，系统说明 GRU 的基本原理、输入输出、训练逻辑、在线推理流程，以及它在 `YOLOv5 + ByteTrack + GRU` 整体链路中的具体作用。

当前系统中的 GRU 不是直接读取图像，也不是直接恢复无人机三维姿态，而是处理已经由检测器和跟踪器形成的二维轨迹序列。

它的核心任务包括：

1. 判断一条连续轨迹是真实无人机还是背景噪声；
2. 根据过去一段时间的运动趋势预测未来数帧二维位置；
3. 为后续目标过滤、轨迹保持和云台前馈控制提供依据；
4. 利用时序信息弥补单帧视觉检测在远距离点目标场景中的不足。

---

# 2. GRU 是什么

GRU 的英文全称为：

```text
Gated Recurrent Unit
```

中文通常翻译为：

```text
门控循环单元
```

GRU 是一种用于处理时间序列的循环神经网络结构。它能够按照时间顺序逐帧读取数据，并在内部保存一个不断更新的“记忆状态”。

在当前无人机项目中，输入不是一张图片，而是一条连续轨迹：

```text
第1帧目标位置
第2帧目标位置
第3帧目标位置
...
第20帧目标位置
```

GRU 会按照顺序依次处理这些轨迹点，并逐步形成对目标运动状态的综合理解。

可以把它理解为：

```text
当前帧观测
    +
过去所有轨迹形成的内部记忆
    ↓
更新当前运动状态
    ↓
继续处理下一帧
```

经过20帧处理后，GRU得到一个压缩后的时序特征向量，用于判断轨迹真假并预测未来位置。

---

# 3. 为什么无人机轨迹需要时序模型

在300米至500米远距离条件下，无人机在图像中可能只剩下数个像素，甚至近似为一个亮点。

此时单帧图像中可以利用的语义信息非常有限，例如：

- 机身轮廓不完整；
- 旋翼不可见；
- 纹理消失；
- 飞行朝向难以判断；
- 检测框尺寸波动明显；
- 单帧置信度较低；
- 容易和树叶、鸟类、反光点混淆。

但是连续多帧中的运动规律仍然存在。

真实无人机通常具有：

- 速度连续性；
- 加速度约束；
- 曲率相对平滑；
- 方向变化具有惯性；
- 短时间内不会完全随机跳变；
- 即使悬停，也存在受控的小幅漂移。

而部分环境干扰可能表现为：

- 高频往复振荡；
- 短时随机跳点；
- 检测框中心不稳定；
- 无持续运动方向；
- 轨迹频繁中断；
- 位置变化和置信度变化无规律。

因此，当前系统需要从“单帧目标检测”升级为“多帧轨迹判定”。

整体关系为：

```text
YOLOv5
负责发现当前帧候选目标
        ↓
ByteTrack
负责把连续帧候选目标连接成Track ID
        ↓
GRU
负责理解该Track ID的时间运动规律
```

---

# 4. GRU 与普通神经网络的区别

## 4.1 普通全连接网络

如果把20帧、每帧8维特征直接展开，可以得到：

```text
20 × 8 = 160维
```

然后送入普通全连接网络。

这种方法虽然也能训练，但它并不天然理解以下关系：

```text
第1帧发生在第2帧之前
第2帧发生在第3帧之前
```

普通网络看到的更像是160个并列数值，而不是一条随时间演化的轨迹。

## 4.2 GRU

GRU会按照时间顺序处理：

```text
x1 → x2 → x3 → ... → x20
```

每处理一帧都会更新内部状态：

```text
x1 → h1
x2 + h1 → h2
x3 + h2 → h3
...
x20 + h19 → h20
```

其中：

```text
xt：第t帧输入特征
ht：处理到第t帧后的隐藏状态
```

最终的 `h20` 表示网络对过去20帧轨迹的综合理解。

---

# 5. GRU 的核心内部结构

GRU主要包含两个门：

```text
更新门
重置门
```

以及一个候选隐藏状态。

## 5.1 更新门

更新门决定：

```text
过去的记忆保留多少
当前的新信息写入多少
```

假设目标已经连续10帧稳定向右运动，但第11帧因为检测误差突然偏移5个像素。

更新门可能认为：

```text
这一帧只是异常检测抖动
不应该完全覆盖之前的稳定运动趋势
```

如果接下来多帧都向新方向运动，GRU才会逐步更新内部记忆，认为目标确实已经转弯。

## 5.2 重置门

重置门决定：

```text
计算当前状态时，需要参考多少过去信息
```

例如：

- 目标从悬停切换为高速平飞；
- 目标突然急转弯；
- 目标经历遮挡后重新出现；
- Track ID发生重新初始化。

此时过去的部分记忆可能不再完全适用，重置门会降低历史状态的影响。

---

# 6. GRU 数学运行逻辑

对于第 `t` 帧输入：

```text
x_t
```

上一时刻隐藏状态：

```text
h_(t-1)
```

GRU首先计算更新门：

```text
z_t = sigmoid(W_z x_t + U_z h_(t-1) + b_z)
```

然后计算重置门：

```text
r_t = sigmoid(W_r x_t + U_r h_(t-1) + b_r)
```

再计算候选状态：

```text
h_candidate =
tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)
```

最后计算当前隐藏状态：

```text
h_t =
(1 - z_t) ⊙ h_(t-1)
+
z_t ⊙ h_candidate
```

其中：

```text
sigmoid：把数值压缩到0到1
tanh：把数值压缩到-1到1
⊙：逐元素乘法
W、U、b：训练过程中自动学习的参数
```

可以直观理解为：

```text
当前隐藏状态
=
保留的历史记忆
+
写入的新运动信息
```

---

# 7. 当前项目中的输入特征

当前基础版本中，每一帧使用8维特征：

```text
x_norm
y_norm
relative_x
relative_y
velocity_x
velocity_y
acceleration_x
acceleration_y
```

## 7.1 归一化位置

```text
x_norm = x_center / image_width
y_norm = y_center / image_height
```

作用：

- 统一不同图像分辨率；
- 减少绝对像素尺寸差异；
- 便于模型在不同相机分辨率之间迁移。

## 7.2 相对位置

```text
relative_x = x_norm[t] - x_norm[0]
relative_y = y_norm[t] - y_norm[0]
```

作用：

- 表示目标相对窗口起点的累积位移；
- 降低不同初始位置的影响；
- 帮助网络判断整体运动方向和净位移。

## 7.3 速度

```text
velocity_x =
(x_norm[t] - x_norm[t-1]) / delta_time

velocity_y =
(y_norm[t] - y_norm[t-1]) / delta_time
```

作用：

- 表示每秒位置变化；
- 区分悬停、匀速运动和快速横穿；
- 帮助识别运动是否连续。

## 7.4 加速度

```text
acceleration_x =
(velocity_x[t] - velocity_x[t-1]) / delta_time

acceleration_y =
(velocity_y[t] - velocity_y[t-1]) / delta_time
```

作用：

- 表示速度变化趋势；
- 帮助识别加速、减速和转弯；
- 用于区分连续机动与随机跳变。

---

# 8. 为什么必须先平滑再计算速度和加速度

远距离目标的检测中心通常存在约1至数个像素的抖动。

如果直接对原始位置求差：

```text
原始坐标
→ 速度
→ 加速度
```

会放大测量误差。

例如真实位置稳定，但检测中心为：

```text
100
101
99
101
100
```

直接计算速度后会出现明显正负振荡，加速度波动更大。

正确流程应为：

```text
原始坐标
→ Savitzky-Golay或卡尔曼平滑
→ 计算速度
→ 计算加速度
```

当前推荐：

```text
平滑窗口：5
多项式阶数：2
```

如果轨迹太短，应自动退化为移动平均或不平滑，不能产生非法结果。

---

# 9. 模型输入 Shape

单条轨迹：

```text
[20, 8]
```

含义：

```text
20帧历史
每帧8个特征
```

批量训练时：

```text
[Batch, 20, 8]
```

例如批大小为64：

```text
[64, 20, 8]
```

表示：

```text
一次处理64条轨迹
每条轨迹20帧
每帧8个特征
```

模型定义示意：

```python
self.gru = nn.GRU(
    input_size=8,
    hidden_size=32,
    num_layers=1,
    batch_first=True,
)
```

---

# 10. GRU 主干的输出

输入：

```text
[B, 20, 8]
```

经过GRU：

```text
sequence_output, hidden = self.gru(x)
```

得到：

```text
sequence_output: [B, 20, 32]
hidden: [1, B, 32]
```

其中：

- `sequence_output[:, t, :]` 表示第t帧对应的隐藏状态；
- `hidden[-1]` 表示最后一层GRU最终隐藏状态。

当前项目通常取：

```python
last_hidden = hidden[-1]
```

Shape为：

```text
[B, 32]
```

这32维向量就是过去20帧轨迹的压缩时序表示。

---

# 11. 多任务输出结构

当前模型不是只做一个任务，而是同时完成：

```text
轨迹真实性分类
+
未来轨迹预测
```

整体结构：

```text
过去20帧轨迹
      ↓
GRU时序编码器
      ↓
32维隐藏状态
      ├─────────────┐
      ▼             ▼
真实性分类头      轨迹预测头
```

---

# 12. 分类分支

分类分支用于判断：

```text
当前Track是否为真实无人机
```

网络结构示例：

```python
self.classifier = nn.Sequential(
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
```

模型直接输出：

```text
classification_logit
```

Shape：

```text
[B, 1]
```

训练时使用：

```python
nn.BCEWithLogitsLoss()
```

推理时转换为概率：

```python
uav_probability = torch.sigmoid(logit)
```

输出示例：

```text
0.93
```

表示模型认为该轨迹有93%的概率属于真实无人机。

---

# 13. 轨迹预测分支

预测分支用于预测未来5帧二维位置变化。

结构示例：

```python
self.predictor = nn.Sequential(
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
```

输出10个数：

```text
5帧 × 2维坐标 = 10
```

reshape后：

```text
[B, 5, 2]
```

表示：

```text
未来第1帧：[dx1, dy1]
未来第2帧：[dx2, dy2]
未来第3帧：[dx3, dy3]
未来第4帧：[dx4, dy4]
未来第5帧：[dx5, dy5]
```

这里预测的是：

```text
相对于历史最后一帧的位移
```

而不是直接预测全局绝对坐标。

优点包括：

- 更容易跨场景；
- 更容易跨图像位置；
- 减少模型记忆绝对画面区域；
- 更适合云台前馈控制。

---

# 14. 当前项目的完整训练样本

每个训练样本包括：

```text
history_features: [20, 8]
future_offsets: [5, 2]
label: 0或1
missing_mask: [20]
episode_id
track_id
motion_type
range_m
noise_profile
```

其中：

```text
history_features
```

来自：

```text
measured_tracks.csv
```

表示带检测误差、漏检、离群点和置信度变化的历史轨迹。

而：

```text
future_offsets
```

来自：

```text
ideal_tracks.csv
```

表示Gazebo理想真值中的未来位置。

核心训练关系为：

```text
过去20帧带噪声观测
        ↓
GRU
        ↓
预测未来5帧理想轨迹
```

这使模型同时具备一定的轨迹去噪能力。

---

# 15. 分类标签

分类标签定义：

```text
1 = 真实无人机
0 = 非无人机
-1 = 未确认
```

常见正样本：

```text
hover
linear
accelerating
decelerating
turn
s_curve
zigzag
approaching
receding
fast_crossing
```

常见负样本：

```text
bird
tree_sway
glint
insect
compression_noise
camera_shake
random_point
```

未确认样本：

```text
label = -1
```

默认不得进入监督训练集。

---

# 16. 联合损失函数

模型同时有分类和预测两个任务。

总损失：

```text
total_loss
=
classification_weight × classification_loss
+
regression_weight × regression_loss
```

## 16.1 分类损失

使用：

```python
classification_loss =
BCEWithLogitsLoss(classification_logits, labels)
```

作用：

- 惩罚把无人机判为噪声；
- 惩罚把噪声判为无人机。

## 16.2 预测损失

使用：

```python
regression_loss =
SmoothL1Loss(predicted_future, target_future)
```

相比普通MSE，SmoothL1对异常跳点更稳定。

## 16.3 只对正样本计算预测损失

鸟类、树叶、反光等负样本不应该被要求学习无人机未来轨迹。

正确逻辑：

```python
positive_mask = labels > 0.5

if positive_mask.any():
    regression_loss = smooth_l1(
        predicted_future[positive_mask],
        target_future[positive_mask],
    )
else:
    regression_loss = zero
```

---

# 17. 模型训练过程

一次训练迭代包含：

```text
读取一批样本
    ↓
移动到训练设备
    ↓
GRU前向传播
    ↓
得到分类Logit和未来轨迹
    ↓
计算分类损失
    ↓
计算正样本轨迹损失
    ↓
合成总损失
    ↓
反向传播
    ↓
梯度裁剪
    ↓
优化器更新参数
```

伪代码：

```python
model.train()

for batch in dataloader:
    features = batch["history_features"].to(device)
    labels = batch["label"].to(device)
    targets = batch["future_offsets"].to(device)

    optimizer.zero_grad()

    logits, predicted_future = model(features)

    cls_loss = classification_loss_fn(
        logits,
        labels,
    )

    positive_mask = labels > 0.5

    if positive_mask.any():
        reg_loss = regression_loss_fn(
            predicted_future[positive_mask],
            targets[positive_mask],
        )
    else:
        reg_loss = torch.zeros(
            (),
            device=device,
        )

    total_loss = (
        cls_weight * cls_loss
        + reg_weight * reg_loss
    )

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()
```

---

# 18. CUDA训练与CPU部署

当前训练可以在CUDA环境中进行，但模型结构本身不能写死CUDA。

训练设备：

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

部署设备：

```python
device = torch.device("cpu")
```

模型保存推荐：

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "preprocess_config": preprocess_config,
        "threshold_config": threshold_config,
    },
    "uav_track_v1_best.pth",
)
```

CPU加载：

```python
checkpoint = torch.load(
    "uav_track_v1_best.pth",
    map_location="cpu",
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.cpu()
model.eval()
```

因此：

```text
CUDA训练
```

不会妨碍后续：

```text
Ubuntu CPU部署
RK3588 CPU部署
ONNX Runtime部署
```

前提是：

- 模型代码不写死 `.cuda()`；
- 保存 `state_dict`；
- 预处理保持一致；
- 部署前完成CUDA和CPU输出一致性测试。

---

# 19. 在线推理逻辑

实际运行时，每个Track ID维护一个轨迹缓存。

例如：

```text
Track 12：
frame 1
frame 2
...
frame 20
```

缓存不足20帧时：

```text
不进行GRU推理
状态保持TENTATIVE
```

达到20帧后：

```text
提取最后20帧
    ↓
预处理
    ↓
构建[1,20,8]
    ↓
GRU前向传播
    ↓
输出概率和未来5帧
```

伪代码：

```python
if track_history.length < 20:
    return "INSUFFICIENT_HISTORY"

features = feature_builder.build(
    track_history.last(20)
)

input_tensor = torch.from_numpy(
    features
).float().unsqueeze(0)

with torch.inference_mode():
    logits, future_offsets = model(
        input_tensor
    )

probability = torch.sigmoid(
    logits
).item()
```

---

# 20. 为什么不能一次低概率就删除轨迹

错误逻辑：

```python
if probability < 0.5:
    delete_track()
```

可能因为一次异常检测或短时遮挡错误删除真实无人机。

正确方式是使用滞回状态机。

状态：

```text
TENTATIVE
CONFIRMED
SUSPICIOUS
SUPPRESSED
LOST
```

推荐逻辑：

```text
概率 >= 0.75
连续3次
→ CONFIRMED

概率 <= 0.25
连续5次
→ SUPPRESSED

0.25 < 概率 < 0.75
→ 保持原状态

连续缺测超过阈值
→ LOST
```

这样可以避免：

- 单次概率抖动；
- CPU与CUDA微小数值差异；
- 临时遮挡；
- YOLO短时低置信度；
- Track中心异常跳点。

---

# 21. 轨迹预测如何用于云台

GRU输出未来5帧相对位置：

```text
[dx1, dy1]
...
[dx5, dy5]
```

可以转换成：

```text
未来像素位置
→ 未来角度偏差
→ 云台前馈控制量
```

控制结构应为：

```text
当前目标位置
→ 反馈控制

未来预测位置
→ 前馈补偿
```

组合：

```text
最终控制量
=
反馈控制量
+
预测置信度 × 前馈控制量
```

只有满足以下条件才使用前馈：

- UAV概率较高；
- 轨迹连续；
- 预测误差较低；
- 无明显相机抖动；
- 云台未达到速度限制；
- 预测位移未超出合理范围。

---

# 22. GRU、YOLOv5与ByteTrack的职责边界

## YOLOv5

输入：

```text
单帧图像
```

输出：

```text
bbox
confidence
class
```

负责回答：

```text
当前帧哪里可能存在无人机
```

## ByteTrack

输入：

```text
连续帧YOLO检测结果
```

输出：

```text
Track ID
轨迹连续关系
```

负责回答：

```text
当前检测框与前一帧哪个目标属于同一对象
```

## GRU

输入：

```text
一个Track ID过去20帧轨迹
```

输出：

```text
真实性概率
未来5帧二维位置
```

负责回答：

```text
这条轨迹是否符合无人机运动规律
接下来可能运动到哪里
```

完整链路：

```text
视频
 ↓
YOLOv5
 ↓
ByteTrack
 ↓
轨迹特征构建
 ↓
GRU
 ↓
状态机
 ↓
告警、目标输出、云台控制
```

---

# 23. 当前Gazebo仿真训练路径

当前系统的仿真训练链路：

```text
PX4 SITL
    ↓
Gazebo产生真实飞行动力学
    ↓
记录三维位置、速度、姿态
    ↓
固定相机投影到二维平面
    ↓
ideal_tracks.csv
    ↓
注入YOLO/跟踪器风格噪声
    ↓
measured_tracks.csv
    ↓
构建20帧历史特征
    ↓
GRU训练
```

训练输入：

```text
measured_tracks过去20帧
```

训练标签：

```text
真实性标签
+
ideal_tracks未来5帧
```

这一阶段还不是真正执行YOLOv5检测，而是通过噪声模型模拟检测器输出。

后续可以增加第二条链路：

```text
Gazebo RGB图像
    ↓
真实YOLOv5
    ↓
ByteTrack
    ↓
真实检测轨迹
    ↓
GRU
```

---

# 24. 不同距离数据对GRU的意义

即使最终输入是二维轨迹，不同距离仍然会改变：

- 目标像素尺寸；
- 每帧像素位移；
- 中心点量化误差占比；
- 检测框波动；
- 置信度；
- 漏检概率；
- 轨迹连续性；
- 速度和加速度可观测性。

因此距离建议作为：

```text
数据生成维度
评估分组维度
噪声模型条件
```

而不一定直接作为GRU输入。

第一版更推荐加入可直接从YOLO获得的特征：

```text
bbox_width_norm
bbox_height_norm
detector_confidence
missing_mask
```

这些特征能够间接反映目标距离和检测质量。

---

# 25. 推荐的增强输入版本

基础版本：

```text
[20, 8]
```

增强版本可扩展为：

```text
[20, 12]
```

每帧特征：

```text
x_norm
y_norm
relative_x
relative_y
velocity_x
velocity_y
acceleration_x
acceleration_y
bbox_width_norm
bbox_height_norm
detector_confidence
missing_mask
```

进一步可加入：

```text
speed
turn_rate
curvature
measurement_uncertainty
delta_time
```

但每增加一个特征，都必须保证：

- Gazebo训练阶段能生成；
- YOLO/ByteTrack部署阶段也能获得；
- 训练和部署计算方式完全一致。

---

# 26. 评估指标

## 26.1 分类指标

```text
Accuracy
Precision
Recall
F1
ROC-AUC
假阳性率
假阴性率
```

在无人机预警系统中，单看Accuracy不够。

重点关注：

```text
Recall
```

因为漏掉真实无人机的代价可能较高。

同时要关注：

```text
False Positive Rate
```

避免树叶、鸟类和反光频繁触发报警。

## 26.2 轨迹指标

### ADE

```text
Average Displacement Error
```

表示未来所有预测帧的平均位置误差。

### FDE

```text
Final Displacement Error
```

表示未来最后一帧的位置误差。

还应按未来时间步统计：

```text
t+1误差
t+2误差
...
t+5误差
```

---

# 27. 常见问题与风险

## 27.1 训练集与测试集泄漏

同一条长轨迹会生成大量重叠窗口。

错误方式：

```text
随机打乱所有窗口
再按8:2划分
```

正确方式：

```text
按episode_id
按完整视频
按飞行任务
```

划分。

## 27.2 只有简单正样本

如果正样本只有平飞，模型可能把：

```text
悬停
急转
S形
遮挡后重现
```

误判为噪声。

## 27.3 负样本过于简单

如果负样本只有零均值正弦树叶运动，模型可能只学习：

```text
有净位移 = 无人机
无净位移 = 树叶
```

必须加入：

- 平滑飞行的鸟类；
- 持续移动的亮点；
- 低频树枝摆动；
- 云台运动造成的背景平移；
- 随机游走噪点。

## 27.4 仿真与真实差异

Gazebo数据适合：

- 验证数据链路；
- 预训练；
- 调参；
- 结构消融；
- 生成大量标签。

但最终仍需要少量真实数据校准：

- 检测噪声；
- YOLO置信度；
- 漏检分布；
- 阈值；
- 实际背景干扰。

---

# 28. 当前推荐网络结构

```python
class UAVTrajectoryNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        future_steps: int = 5,
    ):
        super().__init__()

        self.future_steps = future_steps

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(
                32,
                future_steps * 2,
            ),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "输入必须为[B,T,F]"
            )

        _, hidden = self.gru(x)

        last_hidden = hidden[-1]

        classification_logits = (
            self.classifier(last_hidden)
        )

        future_offsets = self.predictor(
            last_hidden
        )

        future_offsets = future_offsets.view(
            x.shape[0],
            self.future_steps,
            2,
        )

        return (
            classification_logits,
            future_offsets,
        )
```

---

# 29. 推荐在线接口

```python
class TrajectoryInferenceEngine:
    def update_tracks(
        self,
        tracks: list[dict],
    ) -> list[dict]:
        # 输入当前帧的全部Track。
        # 内部维护每个Track ID的历史。
        # 对达到20帧的Track进行Batch推理。
        ...
```

输入示例：

```python
[
    {
        "track_id": 12,
        "timestamp": 3.333,
        "x_center": 1800.2,
        "y_center": 920.6,
        "bbox_width": 4.0,
        "bbox_height": 4.0,
        "image_width": 3840,
        "image_height": 2160,
        "detector_confidence": 0.68,
        "is_missing": False,
    }
]
```

输出示例：

```python
[
    {
        "track_id": 12,
        "uav_probability": 0.91,
        "state": "CONFIRMED",
        "future_offsets": [
            [0.0004, 0.0001],
            [0.0008, 0.0002],
            [0.0012, 0.0003],
            [0.0016, 0.0004],
            [0.0020, 0.0005],
        ],
    }
]
```

---

# 30. 项目最终核心路径

当前系统的完整核心路径可以概括为：

```text
PX4
产生真实动力学运动
    ↓
Gazebo
生成图像和三维真值
    ↓
三维投影
生成理想二维轨迹
    ↓
噪声注入或真实YOLOv5
生成带误差的观测轨迹
    ↓
ByteTrack
维护Track ID
    ↓
特征工程
生成20帧时序输入
    ↓
GRU
提取运动记忆
    ├─────────────┐
    ▼             ▼
真实性分类      未来5帧预测
    ↓             ↓
轨迹状态机      云台前馈控制
```

---

# 31. 最终总结

GRU在当前系统中的本质作用是：

```text
读取一段连续二维轨迹
理解该轨迹的运动规律
判断轨迹是否属于无人机
预测目标下一阶段的位置
```

它并不取代YOLOv5，也不直接读取图像。

三者职责分别为：

```text
YOLOv5：单帧发现目标
ByteTrack：跨帧关联目标
GRU：理解轨迹并预测未来
```

当前模型的核心输入：

```text
过去20帧二维轨迹特征
```

当前模型的核心输出：

```text
真实无人机概率
未来5帧二维相对位置
```

最终训练路径：

```text
measured过去20帧
→ GRU
→ 分类概率 + 未来5帧
→ 与真实性标签和ideal未来轨迹计算损失
→ 反向传播更新模型
```

最终部署路径：

```text
YOLOv5
→ ByteTrack
→ GRU
→ 状态机
→ 告警、过滤或云台控制
```

对于300米至500米点状目标场景，GRU的价值不在于识别外观，而在于利用连续时间中的运动约束，补足单帧视觉信息不足的问题。
