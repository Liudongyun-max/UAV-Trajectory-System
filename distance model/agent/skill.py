#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import tkinter as tk
from pathlib import Path

# 添加项目目录到 Python 搜索路径
distance_model_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(distance_model_dir))

try:
    from distance_model_gui import UAVDistanceStabilizerGUI
except ImportError:
    # 兼容上级目录路径
    sys.path.insert(0, "F:/UAV Trajectory System/distance model")
    from distance_model_gui import UAVDistanceStabilizerGUI

def main():
    parser = argparse.ArgumentParser(description="UAV Trajectory & Distance System Test Automation Skill")
    parser.add_argument(
        "--video", 
        type=str, 
        default="F:/UAV Trajectory System/test  vedio/sky_uav/86246935928d29d8155f867d4e66cb5a.mp4",
        help="Path to the test video file"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["segmentation", "yolov5"], 
        default="segmentation",
        help="Detector mode: 'segmentation' (Image Segmentation + Tracking) or 'yolov5' (YOLOv5 BBox)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="F:/UAV Trajectory System/distance model/models/best.pth",
        help="Path to model weights (.pth or .pt)"
    )
    default_yaml = "F:/UAV Trajectory System/train/configs/distance_gru_static_25f.yaml"
    if not os.path.exists(default_yaml):
        alt_yaml = "F:/UAV Trajectory System/distance model/train/configs/distance_gru_static_25f.yaml"
        if os.path.exists(alt_yaml):
            default_yaml = alt_yaml

    parser.add_argument(
        "--config", 
        type=str, 
        default=default_yaml,
        help="Path to pipeline config yaml"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="F:/UAV Trajectory System/distance model/outputs/reports",
        help="Base report output directory"
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
        
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model weights not found at {model_path}")
        sys.exit(1)
        
    print("==================================================")
    print("      UAV 测距与轨迹预测全流程自动化测试 SKILL")
    print("==================================================")
    print(f"输入视频: {video_path.name}")
    print(f"检测模式: {args.mode}")
    print(f"模型权重: {model_path.name}")
    print(f"输出目录: {args.output_dir}")
    print("==================================================")
    
    # 初始化无窗口的 Tkinter 容器
    root = tk.Tk()
    root.withdraw()
    
    app = UAVDistanceStabilizerGUI(root)
    # 恢复系统的标准输出流
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    # 根据 CLI 设定参数
    app.pure_video_mode_var.set(True)
    app.video_path_var.set(str(video_path))
    app.model_path_var.set(str(model_path))
    app.config_path_var.set(args.config)
    app.output_dir_var.set(args.output_dir)
    
    mode_str = "图像分割+位移追踪" if args.mode == "segmentation" else "YOLOv5 实时目标框定"
    app.detector_mode_var.set(mode_str)
    
    test_result = {"success": False}
    def test_finished_callback(success):
        test_result["success"] = success
        root.destroy()
        
    app.processing_finished = test_finished_callback
    
    print("\n[INFO] 正在启动后台视频处理流（这可能需要 1-2 分钟，请耐心等待）...")
    
    try:
        app.process_worker(
            csv_path="",
            model_path=str(model_path),
            config_path=args.config,
            point_id=f"[CLI-SKILL-{args.mode.upper()}]"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] 运行测试流发生致命异常: {e}")
        root.destroy()
        sys.exit(1)
        
    if not test_result["success"]:
        print("\n[ERROR] 后台处理线程异常终止，测试失败。")
        sys.exit(1)
        
    # 寻找最新生成的 test00x 文件夹
    base_dir = Path(args.output_dir)
    existing_tests = []
    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith("test") and item.name[4:].isdigit():
                try:
                    existing_tests.append(int(item.name[4:]))
                except ValueError:
                    pass
                    
    if not existing_tests:
        print("\n[ERROR] 测试已完成但未找到生成的 test00x 文件夹。")
        sys.exit(1)
        
    latest_idx = max(existing_tests)
    test_folder = base_dir / f"test{latest_idx:03d}"
    summary_md = test_folder / "test_summary.md"
    
    print("\n==================================================")
    print(f"[SUCCESS] 测试执行成功！输出已保存在文件夹: {test_folder.name}")
    print("==================================================")
    
    if summary_md.exists():
        print("\n--- 自动生成的测试分析报告摘要 ---")
        with open(summary_md, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[:40]:
                print(line.rstrip())
            if len(lines) > 40:
                print(f"\n... (其余内容请参阅 {summary_md.name} 文件)")
        print("----------------------------------")
    else:
        print(f"[WARNING] 未能定位到测试报告文档 {summary_md.name}")

if __name__ == "__main__":
    main()
