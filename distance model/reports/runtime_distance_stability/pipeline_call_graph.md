# 实时测距运行链路调用图 (Call Graph)

以下为 `distance_model_gui.py` 内部数据管道的真实调用拓扑关系：

```mermaid
graph TD
    A[视频帧输入: cv2.VideoCapture] -->|cap.read| B(目标检测与图像分割入口)
    
    subgraph 目标提取层
        B -->|YOLOv5 实时目标框定| C1[PyTorchImpl.detect]
        B -->|图像分割+位移追踪| C2[二值化轮廓分析与 Tracker 追踪]
        C1 -->|is_missing=0| D1[提取 bbox 宽高: bw, bh]
        C2 -->|is_missing=0| D1
        C1 -->|is_missing=1| D2[调用上一次有效边界框 fallback]
        C2 -->|is_missing=1| D2
    end
    
    D1 --> E(物理距离计算)
    D2 --> E
    
    subgraph 物理测距层
        E -->|计算公式| F[physics_width/height/area_range_m]
        F -->|物理三项融合| G[fused_range_val]
        G -->|硬上限限制| H[fused_range_val = min/500, fused_range_val]
    end
    
    H --> I(Residual MLP 误差校准)
    
    subgraph MLP 残差预测
        I -->|装配16维特征列| J[predict_mlp]
        J -->|求解残差| K[mlp_predicted_residual_m]
        K -->|绝对合成| L[raw_predicted_range_m = fused_range_val + residual]
    end
    
    L --> M(时序 GRU 状态平滑)
    
    subgraph GRU 时序稳定
        M -->|压入 25 帧滑窗缓存| N{features_queue 长度 < 25?}
        N -->|是| O[直接输出 raw_predicted_range_m]
        N -->|否| P[归一化: mean & std 缩放]
        P -->|GRU前向推理| Q[last_pred, _ = model.forward]
        Q -->|输出绝对测距| R[stable_pred_val = last_pred.item]
    end
    
    R --> S[绘图与 GUI 展示: matplotlib 示波器]
    O --> S
```

### 核心节点文件、函数与变量映射表

| 数据阶段 | 关联文件 | 关联类/函数 | 核心变量 |
| :--- | :--- | :--- | :--- |
| 视频帧读取 | `distance_model_gui.py` | `process_worker` | `cap = cv2.VideoCapture()` |
| 目标检测 / 分割 | `distance_model_gui.py` | `PyTorchImpl.detect` 或 `cv2.findContours` | `detector_mode`, `contours`, `matched_cand` |
| 包围框生成 | `distance_model_gui.py` | `process_worker` 循环体 | `bw`, `bh`, `area_px`, `last_valid_bbox` |
| 物理距离计算 | `distance_model_gui.py` | `process_worker` 循环体 | `physics_width_range_m`, `fused_range_val` |
| MLP 纠偏推理 | `distance_model_gui.py` | `predict_mlp` | `mlp_predicted_residual_m`, `raw_predicted_range_m` |
| GRU 状态缓存 | `distance_model_gui.py` | `process_worker` 循环体 | `features_queue` (限制最大长度为 25) |
| GRU 预测 | `distance_model_gui.py` | `GRUDistanceStabilizer.forward` | `stable_pred_val` |
| 最终 GUI 展示 | `distance_model_gui.py` | `process_worker` 绘图部分 | `history_fused`, `history_stable`, `chart_canvas` |
