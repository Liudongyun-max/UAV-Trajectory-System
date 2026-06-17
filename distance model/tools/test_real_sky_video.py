#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实拍摄视频测距稳定性与平滑过度测试工具。
使用 YOLOv5 和 图像分割 两种检测器在 sky_uav 真实视频上运行，并计算稳定性指标对比。
"""

import sys
import os
import tkinter as tk
from pathlib import Path
import numpy as np
import pandas as pd
import json
import argparse
import warnings

# 压制三方库警告，防止控制台频繁输出而降低渲染效率
warnings.filterwarnings("ignore")

# 解析命令行参数，用于动态限制运行帧数以提高测试效率
parser = argparse.ArgumentParser(description="真实拍摄视频测距防抖性能测试工具")
parser.add_argument("--full", action="store_true", help="是否运行完整视频 (1960帧)")
parser.add_argument("--limit", type=int, default=200, help="限制最大处理帧数，默认 200 帧以加速验证")
args, unknown = parser.parse_known_args()

limit_frames = None if args.full else args.limit

distance_model_dir = Path("F:/UAV Trajectory System/distance model")
sys.path.insert(0, str(distance_model_dir))

from distance_model_gui import UAVDistanceStabilizerGUI

def analyze_stability(diag_dir, mode_name):
    csv_path = diag_dir / "frame_level_comparison.csv"
    events_path = diag_dir / "runtime_events.jsonl"
    
    if not csv_path.exists():
        print(f"[{mode_name}] 错误: frame_level_comparison.csv 未生成")
        return None
        
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
    
    metrics = {}
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
        
        metrics[name] = {
            "负距离次数": neg_count,
            "500m伪饱和次数": fake_500_count,
            "单帧跳变>5m": jump_gt_5,
            "单帧跳变>15m": jump_gt_15,
            "最大帧间跳变 (m)": round(max_jump, 3),
            "P95帧间跳变 (m)": round(p95_jump, 3),
            "输出标准差 (m)": round(std_val, 3)
        }
    return event_counts, metrics

def run_test_for_mode(detector_mode, output_subdir, limit_frames=None):
    print(f"\n======================================================================")
    print(f"   开始运行真实视频测试 - 检测器模式: {detector_mode}")
    print(f"======================================================================")
    
    root = tk.Tk()
    root.withdraw()
    app = UAVDistanceStabilizerGUI(root)
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    # 真实视频路径与输出配置
    video_path = "F:/UAV Trajectory System/test  vedio/sky_uav/86246935928d29d8155f867d4e66cb5a.mp4"
    model_path = "F:/UAV Trajectory System/distance model/models/gru_enhanced.pth"
    config_path = app.config_path_var.get()
    
    output_dir = f"F:/UAV Trajectory System/distance model/outputs/reports/real_sky_test/{output_subdir}"
    
    app.pure_video_mode_var.set(True)
    app.detector_mode_var.set(detector_mode)
    app.video_path_var.set(video_path)
    app.model_path_var.set(model_path)
    app.config_path_var.set(config_path)
    app.output_dir_var.set(output_dir)
    
    execution_success = False
    
    def mock_finished(success):
        nonlocal execution_success
        execution_success = success
        root.destroy()
        
    app.processing_finished = mock_finished
    
    # 对 cv2.VideoCapture.get 进行 Monkeypatch 帧数控制
    import cv2
    original_get = cv2.VideoCapture.get
    
    if limit_frames is not None:
        print(f"   [提示] 已启用 Monkeypatch 帧数限制，最大处理 {limit_frames} 帧。")
        def patched_get(self, propId):
            if propId == cv2.CAP_PROP_FRAME_COUNT:
                return limit_frames
            return original_get(self, propId)
        cv2.VideoCapture.get = patched_get
        
    try:
        # 运行 process_worker
        app.process_worker(
            csv_path="",
            model_path=model_path,
            config_path=config_path,
            point_id=f"[Real Sky - {output_subdir}]"
        )
        
        # 开启 Tkinter 事件循环以等待线程执行完毕
        root.mainloop()
    finally:
        # 恢复原始的 VideoCapture.get
        cv2.VideoCapture.get = original_get
    
    if execution_success:
        diag_dir = Path(output_dir) / "runtime_distance_stability"
        return analyze_stability(diag_dir, detector_mode)
    else:
        print(f"错误: {detector_mode} 模式运行失败。")
        return None

def main():
    global limit_frames
    # 1. 运行图像分割检测器模式
    seg_results = run_test_for_mode("图像分割+位移追踪", "segmentation", limit_frames)
    
    # 2. 运行 YOLOv5 检测器模式
    yolo_results = run_test_for_mode("YOLOv5 实时目标框定", "yolov5", limit_frames)
    
    print("\n" + "="*80)
    print("                      真实拍摄天空视频测距防抖性能测试汇总")
    print("="*80)
    
    modes_results = [
        ("图像分割+位移追踪", seg_results),
        ("YOLOv5 目标定位框定", yolo_results)
    ]
    
    for mode_name, res in modes_results:
        if res is None:
            continue
        evs, metrics = res
        print(f"\n[ 检测模式: {mode_name} ]")
        print(f"  - 短轴塌陷过滤次数:  {evs['SHORT_AXIS_COLLAPSED']}")
        print(f"  - 运动门控创新超限拒绝: {evs['MEASUREMENT_REJECTED']}")
        print(f"  - 几何测距一致性异常降权: {evs['PHYSICS_DISAGREEMENT']}")
        print(f"  - 时序GRU状态重置次数:   {evs['GRU_STATE_RESET']}")
        print(f"  - 负测距值输出拦截次数:   {evs['GRU_OUTPUT_INVALID']}")
        
        print("\n  性能稳定性指标对比:")
        headers = ["测距链路分量", "负距离次数", "最大帧间跳变 (m)", "P95帧间跳变 (m)", "输出标准差 (m)"]
        print(f"  | {' | '.join(headers)} |")
        print(f"  |{'-|' * len(headers)}")
        for ch_name, info in metrics.items():
            row = [
                ch_name,
                str(info["负距离次数"]),
                str(info["最大帧间跳变 (m)"]),
                str(info["P95帧间跳变 (m)"]),
                str(info["输出标准差 (m)"])
            ]
            print(f"  | {' | '.join(row)} |")
            
    print("\n" + "="*80)
    print("测试完毕，所有数据与对比视频已输出至: outputs/reports/real_sky_test/")
    print("="*80)

if __name__ == "__main__":
    main()
