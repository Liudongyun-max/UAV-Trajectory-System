# 技能文档：UAV 实时采集轨迹数据质量分析技能 (UAV Trajectory Data Analysis Skill)

本技能文档定义了一套标准、可复用、且已固化的 UAV 仿真/实时采集数据分析工作流。该工作流对新产生的 Episode 数据进行自动化质量审计，并生成统一格式 of CSV/JSON 数据报告与 19 张可视化图表，评估数据是否可用于模型训练。

---

## 1. 技能概述
* **技能名称**：UAV 数据质量分析技能
* **核心脚本**：
  * [analyze_episode.py](file:///F:/UAV%20Trajectory%20System/analyze_episode.py)：核心单Episode多维度数据质量审计引擎。
  * [batch_analyze.py](file:///F:/UAV%20Trajectory%20System/batch_analyze.py)：自适应递归扫描与Markdown报告批量自动渲染脚本。
* **物理目录重构与镜像对齐**：支持 input 和 outputs 内部多层嵌套飞行姿态子层级（如 `hover/run001/episode_xxxx`）。分析引擎利用 `relpath` 动态计算输入相对相对路径，并在 `export/outputs/analysis/<relative_path>/` 镜像创建输出，实现 100% 对齐。
* **技能目标**：一键生成完整性、高精时间同步（自适应兼容 MAVSDK 与 `mavros_fallback` 降级协议）、自适应物理飞行阶段划分（分离真正的起飞、过渡与 HOVER 阶段）、三轴运动质量、2D 测量噪声及 20+5 训练窗口的量化分析。

---

## 2. 自动化执行指南
### 单Episode分析：
你可通过以下命令对任意嵌套层级的 Episode 目录进行多维度审计：
```bash
D:\anaconda3\python.exe "F:\UAV Trajectory System\analyze_episode.py" -i "F:\UAV Trajectory System\export\input\hover\run001\episode_xxxx"
```

### 批量自适应递归分析与报告自动生成 (推荐)：
你可通过以下命令启动批处理。脚本将自动利用 `os.walk` 递归遍历发现 `export/input/` 下所有包含 `episode.yaml` 的有效 Episode 文件夹，一键运行分析，并自动为每个 Episode 渲染导出专有的中文审计报告与建议书：
```bash
D:\anaconda3\python.exe "F:\UAV Trajectory System\batch_analyze.py"
```

### 命令行参数说明：
* `-i` / `--input`（必填）：指向新 Episode 目录的绝对或相对路径。
* `-o` / `--output`（选填）：指定输出的根目录。若不填，默认将输出写入同级的 `export/outputs/`。

---

## 3. 分析流程与校验指标
自动化脚本运行后，执行以下六大审计步骤：

### 阶段一：数据完整性检查
1. 校验输入目录下的 12 个预期文件是否存在。
2. 校验 `frames`, `truth`, `ideal`, `measured` 四个 CSV 文件行数是否一致，`frame_id` 是否严格连续。
3. 检查数值型字段中是否包含 `NaN` 或 `Inf`。
4. 校验视频 `rgb.mp4` 帧数是否与 CSV 行数一致。
5. 结果自动输出至 `outputs/analysis/episode_xx/integrity_report.json`。

### 阶段二：时间同步质量分析
1. 计算全帧、去首15帧、去首1秒、稳定阶段、可见帧五个维度的 `sync_error_ms` 的均值、中位数、P95/P99 和最大值。
2. 提取并定位同步误差 $> 500\text{ms}$ 的异常帧，保存其前后 10 帧的同步误差详情至 `sync_frame_audit.csv`。
3. 绘制并保存同步误差时序图 (`sync_error_timeline.png`) 和分布直方图 (`sync_error_histogram.png`)。

### 阶段三：物理飞行阶段与可见性划分
1. **自适应时间轴对齐**：遍历 `events.jsonl`，动态寻找第一个包含非空仿真时间戳的事件计算 `t_offset = wall_time - sim_timestamp`，自适应兼容 `MAVSDK` 正常模式与 `mavros_fallback` 降级模式（自动匹配 `ARM`, `RECORD_ACTIVE`, `POST_ROLL` 等事件名）。
2. **纯物理判定自适应切分**：摆脱软件事件时空错位，结合高度 $z \ge 0.1\text{m}$ 判定起飞离地，结合 Z 轴速度 $|v_z| < 0.1\text{m/s}$ 及首次到达 target_z 高度判定稳定，精确分离出 PRE_ROLL (解锁前)、ARM (解锁静止)、TAKEOFF (起飞爬升)、SETTLE (首次到达高度后的 3s 过渡微调) 和 HOVER (超平稳悬停) 阶段。
2. 统计每个阶段的可见帧数、可见率、同步有效率、平均距离和平均像素位置，输出至 `visibility_phase_audit.csv`。
3. 提取最长连续可见帧数、可见/不可见区间列表，绘制飞行阶段高度时序图 (`phase_visibility_timeline.png`)。

### 阶段四：世界运动与悬停质量
1. 基于 `truth.csv` 计算三轴位置、速度、加速度与姿态角。
2. 针对接近悬停高度的 `SETTLE` 阶段，计算水平 RMS 漂移、水平 95% 半径、最大水平偏移、高度标准差、平均水平速度以及净位移。
3. 结果自动输出至 `trajectory_metrics.json`，并生成世界位置、速度、姿态图及 XY 漂移散点图。

### 阶段五：二维理想轨迹与测量噪声
1. 计算 2D 目标理想中心点坐标范围、像素速度、像素加速度以及到画面边界的最小像素距离。
2. 按 `frame_id` 对齐 ideal 和 measured 数据，计算中心平移误差的均值、中位数、标准差、分位数（P95/P99），检查有无系统偏置（Bias X/Y）。
3. 统计单帧漏检比例、连续漏检长度分布、ID Switch 数量与离群点数量。
4. 结果自动输出至 `noise_metrics.json` 并生成测量误差 timeline 图和 XY 轴噪声散点图。

### 阶段六：20+5 训练窗口审计
1. 以窗口长度 25 帧（历史20 + 未来5），使用滑动窗口对所有帧进行审计。
2. 过滤规则：
   * 25帧连续，同一 Track ID，不跨飞行阶段。
   * 未来 5 帧理想真值存在、全部 `visible = true` 且 `sync_valid = true`。
   * 历史 20 帧 `sync_valid` 比例 $\ge 95\%$，有效测量帧数 $\ge 18$，且连续缺测（漏检）不超过 2 帧。
   * 不包含大同步误差异常帧，不包含起飞前或降落后。
3. 分别输出 `valid_windows.csv` 和 `invalid_windows.csv`。
4. 对比 Stride 1, 3, 5 下的有效窗口数量，汇总并输出至 `analysis_summary.json` 中的 `window_audit_summary`。

---

## 4. 输出文件结构与路径
分析结束后，脚本会在输出路径下建立以下标准的镜像层级目录树：

```text
export/outputs/analysis/<relative_path>/    # 多层姿态镜像子目录，如 hover/run001/episode_xxxx/
├── integrity_report.json       # 数据完整性审计报告
├── analysis_summary.json       # Episode 分析元数据与自适应任务类型汇总
├── trajectory_metrics.json     # 三轴位置、速度、偏离及悬停抖动/巡航轨迹跟踪指标
├── noise_metrics.json          # 2D感知测量噪声（中心平移、漏检率、置信度等）统计指标
├── analysis_report.md          # 自动渲染生成的详细中文数据质量分析报告
├── recommendations.md          # 自动渲染生成的中文训练可用性定级建议书
├── sync_frame_audit.csv        # 严重同步误差异常帧前后 10 帧数据审计
├── visibility_phase_audit.csv  # 物理飞行阶段（TAKEOFF, SETTLE, HOVER）帧数及可见性审计
├── window_audit.csv            # 窗口滑动审计状态属性（Stride=1）
├── valid_windows.csv           # 仅包含 VALID 状态的训练切窗列表 (用于 Dataloader 加载)
├── invalid_windows.csv         # 仅包含 INVALID 状态的过滤切窗列表
└── figures/                    # 19张分析图表目录
    ├── sync_error_timeline.png
    ├── sync_error_histogram.png
    ├── phase_visibility_timeline.png
    ├── visibility_timeline.png
    ├── image_position_timeline.png
    ├── world_position.png
    ├── world_velocity.png
    ├── attitude.png
    ├── hover_xy_scatter.png
    ├── ideal_trajectory_xy.png
    ├── ideal_xy_timeline.png
    ├── bbox_size_timeline.png
    ├── pixel_velocity.png
    ├── ideal_measured_overlay.png
    ├── center_error_timeline.png
    ├── center_error_histogram.png
    ├── error_xy_scatter.png
    ├── confidence_timeline.png
    └── missing_timeline.png
```

---

## 5. 训练可用性最终评级规则
分析人员（或 AI Agent）读取 `analysis_summary.json` 后，应依照以下标准确定该 Episode 的训练可用性等级：

* **A级（可进入正式训练）**：
  * 物理阶段 100% 契合 mission 任务设定（存在大量平稳 HOVER 物理帧）。
  * 20+5 窗口审计中，有效窗口比例 $> 80\%$。
  * 平均同步误差 $< 1\text{ms}$，大同步异常帧数为 0。
  * 包含正负样本集。
* **B级（可用于流程验证或预训练）**：
  * 物理阶段大部分契合 mission 任务设定，但包含一定的过渡段。
  * 20+5 窗口审计中，有效窗口比例 $> 50\%$。
  * 剔除预热帧后同步误差均值 $< 2\text{ms}$。
* **C级（清洗后可以使用）**：
  * 物理阶段与 mission 设定存在一定偏离，但可以通过切窗清洗出局部有效的物理阶段数据。
  * 通过 20+5 审计，可清洗出一定数量 of 有效 VALID 窗口供特定阶段（如起飞爬升）的流程跑通、代码冒烟测试与调试。
* **D级（不可进入训练）**：
  * 数据严重缺失，行数不对齐。
  * 同步无效比例 $> 10\%$，或者有效窗口数量为 0。
  * 目标大范围丢失或越界。
