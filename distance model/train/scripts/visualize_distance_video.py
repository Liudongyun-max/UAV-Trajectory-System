#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # 使用无GUI后端，防止在多线程或后台任务中崩溃
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Segoe UI Emoji', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import cv2
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.models.gru_distance_stabilizer import GRUDistanceStabilizer


def find_point_dir(source_root: Path, point_id: str) -> Path | None:
    """递归查找 point_id 对应的目录"""
    for p in source_root.rglob(point_id):
        if p.is_dir() and (p / "point.yaml").exists():
            return p
    return None


def convert_video_path(original_path: str, local_raw_sessions_root: Path) -> Path | None:
    """
    将原始 Linux 视频路径转换为本地 Windows 路径。
    例如将: /home/liu/UAV Trajectory System/uav_gazebo_distance/data/raw_sessions/run017/empty_clean_no_wind/session_xxx/video.mkv
    转换为: {local_raw_sessions_root}/run017/empty_clean_no_wind/session_xxx/video.mkv
    """
    parts = Path(original_path).parts
    try:
        # 寻找 'raw_sessions' 在路径中的索引位置
        idx = parts.index("raw_sessions")
        # 截取后面的部分并与本地根目录拼接
        rel_path = Path(*parts[idx + 1:])
        local_path = local_raw_sessions_root / rel_path
        if local_path.exists():
            return local_path
    except ValueError:
        pass
    
    # 备用：如果包含 session_id，尝试直接在本地根目录下匹配
    p_path = Path(original_path)
    session_id = p_path.parent.name
    # 模糊扫描
    for mkv in local_raw_sessions_root.rglob("video.mkv"):
        if mkv.parent.name == session_id:
            return mkv
            
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time Drone BBox & Distance Comparison Video Visualizer")
    p.add_argument("--config", required=True, help="Path to config YAML")
    p.add_argument("--point-id", default="point_d280_flat_ground_light_wind_rep02", help="指定可视化的Point ID")
    p.add_argument("--fps", type=float, default=25.0, help="输出视频的帧率")
    p.add_argument("--dpi", type=int, default=100, help="折线图渲染DPI")
    args = p.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    out_dir = Path("outputs") / cfg["experiment"]["name"]
    source_root = Path(cfg["data"]["source_root"])  # 通常为 distance model/input/distance_points (或由数据处理导出路径)
    
    # 我们知道实际的 distance model 路径是在 F:/UAV Trajectory System/distance model 下
    distance_model_dir = Path("F:/UAV Trajectory System/distance model")
    local_raw_sessions_root = distance_model_dir / "input" / "raw_sessions"
    
    print(f"Loading global predictions package...")
    # 1. 载入 frame_predictions 数据，用于获取 bounding boxes 和时间戳
    df_all = pd.read_csv(distance_model_dir / "exports" / "gru_input" / "run017" / "frame_predictions.csv")
    df_pt = df_all[df_all["point_id"] == args.point_id].sort_values("frame_id").reset_index(drop=True)
    n_frames = len(df_pt)
    
    if n_frames == 0:
        print(f"Error: Point ID {args.point_id} not found in frame_predictions.csv")
        return
        
    print(f"Found {n_frames} frames for Point {args.point_id}")
    
    # 2. 找到该点的目录以读取 source_reference.json
    point_dir = find_point_dir(distance_model_dir / "input" / "distance_points", args.point_id)
    if not point_dir:
        print(f"Error: Point directory for {args.point_id} not found in distance_points.")
        return
        
    with (point_dir / "source_reference.json").open("r", encoding="utf-8") as f:
        ref_info = json.load(f)
        
    start_frame = ref_info["original_start_frame_id"]
    end_frame = ref_info["original_end_frame_id"]
    orig_video_path = ref_info["original_video_path"]
    
    print(f"Source details: start_frame={start_frame}, end_frame={end_frame}")
    print(f"Original path from Gazebo: {orig_video_path}")
    
    # 3. 寻找本地对应的视频文件
    video_file = convert_video_path(orig_video_path, local_raw_sessions_root)
    if not video_file or not video_file.exists():
        print(f"Warning: Video file not found locally. Searched path: {video_file}")
        print("Falling back to rendering a synthetic background (HUD + Chart only).")
        video_file = None
    else:
        print(f"Found local video file: {video_file}")
        
    # 4. 载入 GRU 模型并做序列预测 (在线滑动窗口推理，与 visualize_distance_stabilizer.py 一致)
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
    
    # 预计算整个时序上的 GRU 平滑输出
    stable_predictions = np.zeros(n_frames, dtype=np.float32)
    seq_len = 25
    raw_preds = df_pt["raw_predicted_range_m"].to_numpy(dtype=np.float32)
    stable_predictions[:seq_len-1] = raw_preds[:seq_len-1]
    
    for i in range(seq_len - 1, n_frames):
        sub_df = df_pt.iloc[i - seq_len + 1 : i + 1]
        x_raw = sub_df[feature_cols].to_numpy(dtype=np.float32)
        x_norm = (x_raw - means) / stds
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            last_pred, _ = model(x_t)
        stable_predictions[i] = last_pred.item()
        
    # 5. 设置 OpenCV VideoCapture (若视频文件可用)
    cap = None
    if video_file:
        cap = cv2.VideoCapture(str(video_file))
        # 定位到对应的起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
    # 6. 初始化合成画面的尺寸
    # 视频帧缩放到 960x540，Matplotlib 图表渲染成 720x540，组合分辨率为 1680x540
    video_w, video_h = 960, 540
    chart_w, chart_h = 720, 540
    out_w, out_h = video_w + chart_w, video_h
    
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = reports_dir / f"visualized_distance_{args.point_id}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(out_video_path), fourcc, args.fps, (out_w, out_h))
    
    print(f"Output video path: {out_video_path}")
    print(f"Canvas size: {out_w}x{out_h} ({video_w}x{video_h} video + {chart_w}x{chart_h} oscilloscope chart)")
    
    # 预设 Matplotlib 图表
    plt.ioff()
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=args.dpi)
    
    timestamps = df_pt["timestamp"].to_numpy()
    time_s = timestamps - timestamps[0]  # 首帧对齐到 0s
    true_ranges = df_pt["true_range_m"].to_numpy()
    fused_ranges = df_pt["fused_range_m"].to_numpy()
    
    min_dist = min(true_ranges.min(), fused_ranges.min(), stable_predictions.min())
    max_dist = max(true_ranges.max(), fused_ranges.max(), stable_predictions.max())
    dist_span = max_dist - min_dist
    y_limits = (min_dist - 0.15 * dist_span - 1.0, max_dist + 0.15 * dist_span + 1.0)
    
    snapshot_frames = [int(n_frames * 0.2), int(n_frames * 0.5), int(n_frames * 0.8)]
    
    # 7. 逐帧读取与拼接写入
    for i in range(n_frames):
        # 7.1 读取/生成左半边视频帧
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Video stream ended prematurely at frame index {i}")
                # 后面帧使用黑色背景
                frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        else:
            # 仿真背景：全黑画布
            frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
            
        # 缩放到 960x540
        orig_h, orig_w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (video_w, video_h))
        
        # 坐标缩放因子
        scale_x = video_w / orig_w
        scale_y = video_h / orig_h
        
        # 7.2 绘制目标检测框与悬浮标注
        row = df_pt.iloc[i]
        is_missing = int(row["is_missing"])
        is_interpolated = int(row["is_interpolated"])
        
        if is_missing == 0:
            x1 = int(row["bbox_x1"] * scale_x)
            y1 = int(row["bbox_y1"] * scale_y)
            x2 = int(row["bbox_x2"] * scale_x)
            y2 = int(row["bbox_y2"] * scale_y)
            
            box_color = (0, 165, 255) if is_interpolated else (0, 255, 255)  # 插值框用橙色，真实框用黄色
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), box_color, 2)
            
            # 悬浮标注测距值
            label = f"Est: {stable_predictions[i]:.2f}m"
            cv2.putText(frame_resized, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)
            
        # 7.3 绘制 HUD 半透明背景与状态面板
        overlay = frame_resized.copy()
        cv2.rectangle(overlay, (15, 15), (320, 165), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame_resized, 0.4, 0, frame_resized)
        
        # 写入 HUD 文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        curr_time = time_s[i]
        cv2.putText(frame_resized, f"Point: {args.point_id}", (25, 35), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_resized, f"Time: {curr_time:.2f}s | Frame: {i}/{n_frames}", (25, 58), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame_resized, f"True Range:  {true_ranges[i]:.3f} m", (25, 83), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_resized, f"Physics Fused: {fused_ranges[i]:.3f} m", (25, 105), font, 0.42, (255, 120, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_resized, f"GRU Stabilized: {stable_predictions[i]:.3f} m", (25, 128), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        
        # 显示误差
        err = abs(stable_predictions[i] - true_ranges[i])
        cv2.putText(frame_resized, f"Abs Error: {err:.3f} m", (25, 150), font, 0.42, (0, 0, 255) if err > 0.1 else (0, 255, 0), 1, cv2.LINE_AA)
        
        # 标出异常帧状态
        if is_missing == 1:
            cv2.putText(frame_resized, "[MISSING FRAME]", (20, video_h - 20), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        elif is_interpolated == 1:
            cv2.putText(frame_resized, "[INTERPOLATED FRAME]", (20, video_h - 20), font, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
            
        # 7.4 用 Matplotlib 渲染右侧动态图表 (随着当前帧前进而更新)
        ax.clear()
        
        # 绘制历史整段曲线，让垂直指示线随着时间走
        ax.plot(time_s[:i+1], true_ranges[:i+1], color="green", linestyle="-", linewidth=2.0, label="True Range")
        ax.plot(time_s[:i+1], fused_ranges[:i+1], color="blue", linestyle="--", alpha=0.6, label="Physics Fused")
        ax.plot(time_s[:i+1], stable_predictions[:i+1], color="red", linestyle="-", linewidth=1.8, label="GRU Stabilizer")
        
        # 绘制垂直指示线 (当前时间)
        ax.axvline(x=curr_time, color="black", linestyle=":", linewidth=1.5)
        ax.plot([curr_time], [stable_predictions[i]], "ro")  # 当前预测红点
        
        ax.set_title("测距曲线实时对比 (Oscilloscope)", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("相对时间 (秒)", fontsize=9)
        ax.set_ylabel("距离估算值 (米)", fontsize=9)
        ax.set_xlim(0, time_s[-1])
        ax.set_ylim(y_limits)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)
        
        # 将 matplotlib canvas 转成 numpy image
        fig.tight_layout()
        fig.canvas.draw()
        
        # 提取 RGBA 缓冲区，转为 OpenCV 的 BGR 图像
        chart_rgba = np.asarray(fig.canvas.buffer_rgba())
        chart_bgr = cv2.cvtColor(chart_rgba, cv2.COLOR_RGBA2BGR)
        # 缩放到 720x540
        chart_resized = cv2.resize(chart_bgr, (chart_w, chart_h))
        
        # 7.5 拼接左右画面
        combined_canvas = np.hstack((frame_resized, chart_resized))
        
        # 写入视频
        out_video.write(combined_canvas)
        
        # 保存特定帧快照用于文档
        if i in snapshot_frames:
            snapshot_path = reports_dir / f"distance_snapshot_{args.point_id}_frame_{i}.png"
            cv2.imwrite(str(snapshot_path), combined_canvas)
            print(f"Saved snapshot to: {snapshot_path}")
            
    # 收尾
    if cap is not None:
        cap.release()
    out_video.release()
    plt.close(fig)
    print(f"Visualized video generated successfully at: {out_video_path}")


if __name__ == "__main__":
    main()
