#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV 测距稳定性评估与指标对比报告生成工具。
分析 frame_level_comparison.csv 与 runtime_events.jsonl，生成多通道测距对比指标。
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    root_dir = Path("F:/UAV Trajectory System/distance model")
    diag_dir = root_dir / "outputs/reports/runtime_distance_stability"
    csv_path = diag_dir / "frame_level_comparison.csv"
    events_path = diag_dir / "runtime_events.jsonl"
    
    if not csv_path.exists():
        print(f"Error: frame_level_comparison.csv not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    events = []
    if events_path.exists():
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
                    
    event_counts = {
        "SHORT_AXIS_COLLAPSED": 0,
        "MEASUREMENT_REJECTED": 0,
        "MASK_FRAGMENTED": 0,
        "TARGET_LOST": 0,
        "TARGET_REACQUIRED": 0,
        "GRU_STATE_RESET": 0,
        "GRU_OUTPUT_INVALID": 0,
        "PHYSICS_DISAGREEMENT": 0
    }
    
    for ev in events:
        etype = ev.get("event_type")
        if etype in event_counts:
            event_counts[etype] += 1
            
    channels = {
        "Original Physics": "raw_physics_range_m",
        "Robust Physics": "robust_physics_range_m",
        "Residual MLP": "mlp_range_m",
        "Kalman": "kalman_range_m",
        "GRU": "gru_range_m",
        "GRU + Adaptive Kalman": "final_stable_range_m"
    }
    
    metrics_summary = {}
    
    for name, col in channels.items():
        if col not in df.columns:
            continue
            
        series = df[col].to_numpy(dtype=np.float32)
        valid_series = series[~np.isnan(series)]
        
        neg_count = int(np.sum(valid_series < 0.0))
        fake_500_count = int(np.sum((valid_series >= 499.0) & (valid_series <= 501.0)))
        
        if len(valid_series) > 1:
            diffs = np.abs(np.diff(valid_series))
            max_jump = float(np.max(diffs))
            p95_jump = float(np.percentile(diffs, 95))
            jump_gt_5 = int(np.sum(diffs > 5.0))
            jump_gt_15 = int(np.sum(diffs > 15.0))
        else:
            max_jump, p95_jump, jump_gt_5, jump_gt_15 = 0.0, 0.0, 0, 0
            
        std_val = float(np.std(valid_series)) if len(valid_series) > 0 else 0.0
        
        coasting_duration = 0.0
        if "tracking_state" in df.columns and "frame_delta_t_s" in df.columns:
            coasting_rows = df[(df["tracking_state"] == "COASTING") & (df["measurement_valid"] == 0)]
            coasting_duration = float(coasting_rows["frame_delta_t_s"].sum())
            
        metrics_summary[name] = {
            "负距离输出次数": neg_count,
            "500m伪饱和次数": fake_500_count,
            "单帧跳变>5m次数": jump_gt_5,
            "单帧跳变>15m次数": jump_gt_15,
            "最大帧间跳变 (米)": round(max_jump, 3),
            "P95帧间跳变 (米)": round(p95_jump, 3),
            "输出距离标准差 (米)": round(std_val, 3),
            "丢失后错误旧值保持时间 (秒)": round(coasting_duration, 3) if name in ["GRU", "GRU + Adaptive Kalman"] else "N/A"
        }
        
    out_json = {
        "event_counts": event_counts,
        "channel_metrics": metrics_summary
    }
    
    with open(diag_dir / "before_after_metrics.json", "w", encoding="utf-8") as jf:
        json.dump(out_json, jf, indent=2, ensure_ascii=False)
        
    print("\n" + "="*80)
    print("                      UAV 实时测距优化前后对比报告")
    print("="*80)
    
    print("\n[ 系统级异常事件统计 ]")
    print(f"  - 短轴塌陷检测拦截次数 (SHORT_AXIS_COLLAPSED):  {event_counts['SHORT_AXIS_COLLAPSED']}")
    print(f"  - 运动门控创新值超限拒绝次数 (MEASUREMENT_REJECTED): {event_counts['MEASUREMENT_REJECTED']}")
    print(f"  - 轮廓Solidity碎裂降权次数 (MASK_FRAGMENTED):    {event_counts['MASK_FRAGMENTED']}")
    print(f"  - 目标进入丢失状态次数 (TARGET_LOST):          {event_counts['TARGET_LOST']}")
    print(f"  - 重新捕获成功次数 (TARGET_REACQUIRED):         {event_counts['TARGET_REACQUIRED']}")
    print(f"  - GRU时序隐藏状态重置次数 (GRU_STATE_RESET):     {event_counts['GRU_STATE_RESET']}")
    print(f"  - GRU负距离异常输出拦截次数 (GRU_OUTPUT_INVALID): {event_counts['GRU_OUTPUT_INVALID']}")
    print(f"  - 几何候选测距一致性异常降权次数 (PHYSICS_DISAGREEMENT): {event_counts['PHYSICS_DISAGREEMENT']}")
    
    print("\n[ 多通道测距链路性能指标对比 ]")
    headers = ["评估指标", "Original Physics", "Robust Physics", "Residual MLP", "Kalman", "GRU", "GRU + Adaptive Kalman"]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-|' * len(headers)}")
    
    row_keys = [
        ("负距离输出次数", "负距离输出次数"),
        ("500m伪饱和次数", "500m伪饱和次数"),
        ("单帧跳变>5m次数", "单帧跳变>5m次数"),
        ("单帧跳变>15m次数", "单帧跳变>15m次数"),
        ("最大帧间跳变 (米)", "最大帧间跳变 (米)"),
        ("P95帧间跳变 (米)", "P95帧间跳变 (米)"),
        ("输出距离标准差 (米)", "输出距离标准差 (米)"),
        ("丢失后错误旧值保持时间 (秒)", "丢失后错误旧值保持时间 (秒)")
    ]
    
    for label, key in row_keys:
        cells = [label]
        for name in headers[1:]:
            val = metrics_summary.get(name, {}).get(key, "N/A")
            cells.append(str(val))
        print(f"| {' | '.join(cells)} |")
        
    print("="*80)
    print(f"指标统计完成！已将 before_after_metrics.json 导出至: {diag_dir.name}")

if __name__ == "__main__":
    main()
