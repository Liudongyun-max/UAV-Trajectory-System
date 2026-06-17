#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import _bootstrap  # noqa: F401
from training.models.gru_distance_stabilizer import GRUDistanceStabilizer
from training.utils.distance_metrics import apply_kalman_filter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--point-id", default=None, help="指定可视化的Point ID, 留空则随机选择一个")
    args = p.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    out_dir = Path("outputs") / cfg["experiment"]["name"]
    source_root = Path(cfg["data"]["source_root"])
    
    # 1. 载入 frame_predictions 数据
    df_all = pd.read_csv(source_root / "frame_predictions.csv")
    
    # 2. 选定 Point
    if args.point_id:
        point_id = args.point_id
    else:
        # 随机选择一个含有至少 120 帧的 Point
        point_counts = df_all["point_id"].value_counts()
        point_id = point_counts.index[0]
        
    df_pt = df_all[df_all["point_id"] == point_id].sort_values("frame_id").reset_index(drop=True)
    n_frames = len(df_pt)
    print(f"Visualizing Point ID: {point_id} (contains {n_frames} frames)")
    
    # 3. 载入 GRU 模型并做序列预测
    with (out_dir / "feature_schema.json").open("r", encoding="utf-8") as f:
        schema = json.load(f)
    feature_cols = schema["feature_columns"]
    input_dim = schema["input_dim"]
    
    with (out_dir / "normalization.json").open("r", encoding="utf-8") as f:
        norm_info = json.load(f)
    means = np.array([norm_info["mean"][c] for c in feature_cols], dtype=np.float32)
    stds = np.array([norm_info["std"][c] for c in feature_cols], dtype=np.float32)
    
    model = GRUDistanceStabilizer(input_dim=input_dim)
    model.load_state_dict(torch.load(out_dir / "checkpoints" / "best.pth", map_location="cpu"))
    model.eval()
    
    # 对全量帧进行滑动窗口 GRU 推理，平滑输出
    stable_predictions = np.zeros(n_frames, dtype=np.float32)
    
    # 开头 sequence_length-1 帧由于窗口历史不够，用 raw_predicted_range_m 代替作为预热
    seq_len = 25
    raw_preds = df_pt["raw_predicted_range_m"].to_numpy(dtype=np.float32)
    stable_predictions[:seq_len-1] = raw_preds[:seq_len-1]
    
    # 逐帧在线滑动窗口推理
    for i in range(seq_len - 1, n_frames):
        # 取前 25 帧构成当前窗口
        sub_df = df_pt.iloc[i - seq_len + 1 : i + 1]
        x_raw = sub_df[feature_cols].to_numpy(dtype=np.float32)
        x_norm = (x_raw - means) / stds
        
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0) # [1, 25, input_dim]
        with torch.no_grad():
            last_pred, _ = model(x_t)
        stable_predictions[i] = last_pred.item()

    # 4. 计算卡尔曼滤波平滑作为传统滤波对比
    missing_mask = df_pt["is_missing"].to_numpy(dtype=np.float32)
    kalman_predictions = apply_kalman_filter(raw_preds, missing_mask, q=0.01, r=0.5)

    # 5. 绘制可视化折线图
    timestamps = df_pt["timestamp"].to_numpy()
    # 将时间戳以首帧对齐到 0s 起点
    time_s = timestamps - timestamps[0]
    
    true_range = df_pt["true_range_m"].to_numpy()
    fused_range = df_pt["fused_range_m"].to_numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 子图 1: 测距曲线
    ax1.plot(time_s, true_range, label="Ground Truth (真实距离)", color="green", linewidth=2.5)
    ax1.plot(time_s, fused_range, label="Physics Fused (物理融合基线)", color="blue", linestyle="--", alpha=0.7)
    ax1.plot(time_s, raw_preds, label="MLP Corrected (MLP 单帧校正)", color="orange", alpha=0.6)
    ax1.plot(time_s, kalman_predictions, label="MLP + Kalman Filter", color="gold", linestyle="-.", alpha=0.8)
    ax1.plot(time_s, stable_predictions, label="MLP + GRU Stabilizer (时序稳定生产)", color="red", linewidth=2)
    
    ax1.set_title(f"Point {point_id} 测距稳定与平滑曲线对比", fontsize=14, pad=12)
    ax1.set_ylabel("测距距离 (m)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=10)
    
    # 子图 2: 测距绝对误差
    ax2.plot(time_s, np.abs(fused_range - true_range), label="Physics Fused 误差", color="blue", linestyle="--", alpha=0.7)
    ax2.plot(time_s, np.abs(raw_preds - true_range), label="MLP Corrected 误差", color="orange", alpha=0.6)
    ax2.plot(time_s, np.abs(kalman_predictions - true_range), label="Kalman Filter 误差", color="gold", linestyle="-.", alpha=0.8)
    ax2.plot(time_s, np.abs(stable_predictions - true_range), label="GRU Stabilizer 误差", color="red", linewidth=2)
    
    ax2.set_title("各算法绝对测距误差对比", fontsize=12)
    ax2.set_xlabel("时间 (s)", fontsize=12)
    ax2.set_ylabel("绝对误差 (m)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10)
    
    # 标注漏检区间 (is_missing == 1)
    missing_intervals = np.where(missing_mask == 1)[0]
    if len(missing_intervals) > 0:
        # 寻找连续区间并画背景阴影
        for idx in missing_intervals:
            ax1.axvspan(time_s[idx], time_s[idx] + 0.04, color="gray", alpha=0.15)
            ax2.axvspan(time_s[idx], time_s[idx] + 0.04, color="gray", alpha=0.15)
        # 只在图例上标注一次
        ax1.text(time_s[missing_intervals[0]], ax1.get_ylim()[0] + 0.5, "灰色阴影区域为漏检帧", color="dimgray", fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / "reports" / "distance_stabilization_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Visualization plot saved successfully to: {plot_path}")


import yaml
if __name__ == "__main__":
    main()
