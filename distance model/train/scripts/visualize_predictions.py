"""
无人机轨迹预测可视化与噪声抑制效果验证脚本

用法:
    python scripts/visualize_predictions.py \
        --model models/gru_baseline_final.pth \
        --episode F:/UAV Trajectory System/export/input/run004/zigzag/episode_zigzag_d050_vmedium_forest_edge_medium_seed0273 \
        --smooth True
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.models.gru_trajectory import UAVTrajectoryNet
from training.utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize UAV Trajectory Predictions")
    parser.add_argument("--model", type=str, default="models/gru_baseline_final.pth", help="Path to trained model")
    parser.add_argument("--episode", type=str, required=True, help="Path to episode folder")
    parser.add_argument("--smooth", type=str, default="True", help="Apply Savitzky-Golay pre-smoothing (True/False)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="C:/Users/FUN-C/.gemini/antigravity/brain/e2cbd756-8512-4d1d-90cd-3f01b80d2f02", help="Image output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    
    # 1. Load Model
    model = UAVTrajectoryNet.load_from_checkpoint(args.model, device=str(device))
    model.eval()
    print(f"Model loaded: {args.model} on {device}")
    
    # Read config from checkpoint
    checkpoint = torch.load(args.model, map_location="cpu")
    model_config = checkpoint.get("model_config", {})
    input_dim = model_config.get("input_dim", model.input_dim)
    future_steps = model_config.get("future_steps", model.future_steps)
    seq_len = model_config.get("seq_len", 20)
    
    # 2. Read Data
    ideal_csv = os.path.join(args.episode, "ideal_tracks.csv")
    measured_csv = os.path.join(args.episode, "measured_tracks.csv")
    video_path = os.path.join(args.episode, "rgb.mp4")
    
    if not (os.path.exists(ideal_csv) and os.path.exists(measured_csv) and os.path.exists(video_path)):
        print(f"Error: Missing ideal_tracks.csv, measured_tracks.csv, or rgb.mp4 in {args.episode}")
        return
        
    df_ideal = pd.read_csv(ideal_csv)
    df_measured = pd.read_csv(measured_csv)
    
    # Fill NaN measurements
    df_measured = df_measured.ffill().bfill()
    
    # Create smooth copy if requested
    smooth_active = args.smooth.lower() == "true"
    df_measured_processed = df_measured.copy()
    
    if smooth_active:
        from scipy.signal import savgol_filter
        window = 5
        polyorder = 2
        if len(df_measured_processed) >= window:
            df_measured_processed["x_center"] = savgol_filter(df_measured_processed["x_center"].values.astype(np.float64), window, polyorder)
            df_measured_processed["y_center"] = savgol_filter(df_measured_processed["y_center"].values.astype(np.float64), window, polyorder)
            print("Savitzky-Golay filtering applied to measurements.")
            
    # Find the first frame where the drone becomes visible
    visible_indices = df_ideal[df_ideal["visible"] == 1].index
    start_idx = visible_indices[0] if len(visible_indices) > 0 else 0
    print(f"Drone visible starting from frame index: {start_idx}")
            
    # 3. Setup Video Processing
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_video_path = os.path.join(args.output_dir, "visualized_trajectory.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps} fps, total {total_frames} frames.")
    print(f"Output video will be saved to: {output_video_path}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    frame_idx = 0
    saved_frames = [350, 450, 550]  # Frame IDs to save as images
    last_valid_offset = (0.0, 0.0) # 闭环检测校正偏移量
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 图像实时处理：提取深色无人机在当前帧视频中的质心坐标，进行校准对准
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 天空背景为灰白色，无人机机体为深黑色，通过二值化可 100% 纯净分割
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        drone_center = None
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            # 过滤小噪点和过大背景
            if 5 < area < 2000:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # 避开左上角状态文字区域和右上角图例区域，允许检测边缘地面
                    in_text_area = (cX < 350 and cY < 250)
                    in_legend_area = (cX > width - 300 and cY < 250)
                    if not (in_text_area or in_legend_area):
                        if area > max_area:
                            max_area = area
                            drone_center = (cX, cY)
                            
        # 获取当前帧的 CSV 绝对像素坐标（1280x720 尺度下）
        current_x = df_measured.iloc[frame_idx]["x_center"]
        current_y = df_measured.iloc[frame_idx]["y_center"]
        
        # 闭环更新图像与数据的系统性平移偏差
        if drone_center is not None:
            last_valid_offset = (drone_center[0] - current_x, drone_center[1] - current_y)
            
        # Draw on frame only if we are past start_idx and have sufficient history
        if frame_idx >= max(seq_len, start_idx) and frame_idx < len(df_measured) - future_steps:
            # Get historical windows (seq_len frames)
            history_rows = df_measured_processed.iloc[frame_idx - seq_len + 1 : frame_idx + 1]
            
            # Compute features for model (模型在 2560x1440 归一化特征下进行前向推理，保持预测逻辑的原始设定)
            x = history_rows["x_center"].values.astype(np.float64)
            y = history_rows["y_center"].values.astype(np.float64)
            
            x_norm = x / 2560.0
            y_norm = y / 1440.0
            
            relative_x = x_norm - x_norm[0]
            relative_y = y_norm - y_norm[0]
            
            velocity_x = np.diff(x_norm, prepend=x_norm[0])
            velocity_y = np.diff(y_norm, prepend=y_norm[0])
            
            acceleration_x = np.diff(velocity_x, prepend=velocity_x[0])
            acceleration_y = np.diff(velocity_y, prepend=velocity_y[0])
            
            feature_list = [
                x_norm, y_norm, relative_x, relative_y,
                velocity_x, velocity_y, acceleration_x, acceleration_y
            ]
            
            if input_dim == 12:
                w = history_rows["bbox_width"].values.astype(np.float64)
                h = history_rows["bbox_height"].values.astype(np.float64)
                w_norm = w / 2560.0
                h_norm = h / 1440.0
                aspect_ratio = w_norm / (h_norm + 1e-8)
                conf = history_rows["detector_confidence"].values.astype(np.float64) if "detector_confidence" in history_rows.columns else np.ones_like(x)
                feature_list.extend([w_norm, h_norm, aspect_ratio, conf])
                
            features_np = np.stack(feature_list, axis=-1).astype(np.float32)
            features_t = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                logits, pred_offsets = model(features_t)
                pred_offsets = pred_offsets.squeeze(0).cpu().numpy()  # [5, 2]
                classification_prob = torch.sigmoid(logits).item()
                
            # Determine prediction coordinates
            # 使用模型输出的比例偏移，缩放到 1280x720 图像位移空间，并加上闭环校正偏差
            pred_coords = []
            for offset in pred_offsets:
                px = current_x + offset[0] * 1280.0
                py = current_y + offset[1] * 720.0
                pred_coords.append((int(px + last_valid_offset[0]), int(py + last_valid_offset[1])))
                
            # Get Ground Truth future coords (应用闭环对齐)
            gt_rows = df_ideal.iloc[frame_idx + 1 : frame_idx + 1 + future_steps]
            gt_coords = [(int(r["x_center"] + last_valid_offset[0]), int(r["y_center"] + last_valid_offset[1])) 
                         for _, r in gt_rows.iterrows()]
            
            # Get historical coordinates (应用闭环对齐)
            hist_coords_processed = [(int(df_measured_processed.iloc[i]["x_center"] + last_valid_offset[0]), int(df_measured_processed.iloc[i]["y_center"] + last_valid_offset[1])) 
                                     for i in range(frame_idx - seq_len + 1, frame_idx + 1)]
            hist_coords_raw = [(int(df_measured.iloc[i]["x_center"] + last_valid_offset[0]), int(df_measured.iloc[i]["y_center"] + last_valid_offset[1])) 
                               for i in range(frame_idx - seq_len + 1, frame_idx + 1)]
            
            # Draw raw noisy history (thin red line)
            for i in range(1, len(hist_coords_raw)):
                cv2.line(frame, hist_coords_raw[i-1], hist_coords_raw[i], (0, 0, 200), 1)
                
            # Draw processed history (thick red/orange line)
            for i in range(1, len(hist_coords_processed)):
                cv2.line(frame, hist_coords_processed[i-1], hist_coords_processed[i], (0, 69, 255), 2)
                
            # Draw Ground Truth future (green)
            for i in range(1, len(gt_coords)):
                cv2.line(frame, gt_coords[i-1], gt_coords[i], (0, 255, 0), 2)
            for coord in gt_coords:
                cv2.circle(frame, coord, 4, (0, 255, 0), -1)
                
            # Draw Predictions future (cyan)
            for i in range(1, len(pred_coords)):
                cv2.line(frame, pred_coords[i-1], pred_coords[i], (255, 255, 0), 2)
            for coord in pred_coords:
                cv2.circle(frame, coord, 4, (255, 255, 0), -1)
                
            # Calculate pixel ADE for this frame in original 1280x720 coordinate space (基于对齐后的实际屏幕位置计算，完全反映真实观感下的像素误差)
            errors = []
            for pred, gt in zip(pred_coords, gt_coords):
                errors.append(np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2))
            frame_ade = np.mean(errors) if errors else 0.0
            
            # Draw target bounding box (黄色检测框对准无人机中心，无任何尺寸裁剪和拉伸)
            curr_row = df_measured.iloc[frame_idx]
            w = int(curr_row.get("bbox_width", 50))
            h = int(curr_row.get("bbox_height", 50))
            cx_drawn = int(current_x + last_valid_offset[0])
            cy_drawn = int(current_y + last_valid_offset[1])
            cv2.rectangle(frame, (cx_drawn - w//2, cy_drawn - h//2), (cx_drawn + w//2, cy_drawn + h//2), (0, 255, 255), 2)
            
            # Text information
            is_noise = classification_prob < 0.5
            cls_text = "ANOMALY/NOISE" if is_noise else "TRUE UAV"
            cls_color = (0, 0, 255) if is_noise else (0, 255, 0)
            
            # 创建半透明 HUD 背景遮罩
            overlay = frame.copy()
            # 左上角状态面板背景
            cv2.rectangle(overlay, (20, 20), (420, 185), (20, 20, 20), -1)
            # 右上角图例面板背景
            cv2.rectangle(overlay, (width - 330, 20), (width - 20, 185), (20, 20, 20), -1)
            # 融合半透明遮罩
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            
            # 绘制精致的状态文本 (fontScale=0.52, thickness=1, LINE_AA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (40, 55), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "State: ", (40, 95), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            # 将 TRUE UAV 用高亮颜色独立绘制
            cv2.putText(frame, cls_text, (100, 95), font, 0.52, cls_color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"(Prob: {classification_prob:.4f})", (240, 95), font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            
            cv2.putText(frame, f"Frame ADE: {frame_ade:.2f} px", (40, 135), font, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Smooth status: {'Savgol ON' if smooth_active else 'OFF'}", (40, 170), font, 0.52, (255, 165, 0), 1, cv2.LINE_AA)
            
            # 绘制右上角图例
            cv2.putText(frame, "Legend:", (width - 310, 55), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 绘制图例包含对应的彩色代表线段
            # - Noisy BBox Input (橙红色)
            cv2.line(frame, (width - 310, 90), (width - 270, 90), (0, 69, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Noisy BBox Input", (width - 250, 95), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            
            # - Prediction Future (青色)
            cv2.line(frame, (width - 310, 125), (width - 270, 125), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Prediction Future (5 steps)", (width - 250, 130), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            
            # - Ground Truth Future (绿色)
            cv2.line(frame, (width - 310, 160), (width - 270, 160), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Ground Truth Future (5 steps)", (width - 250, 165), font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            
        out.write(frame)
        
        # Save specific frames as images for documentation
        if frame_idx in saved_frames:
            img_path = os.path.join(args.output_dir, f"visualized_frame_{frame_idx}.png")
            cv2.imwrite(img_path, frame)
            print(f"Saved snapshot to {img_path}")
            
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Trajectory visualization generation complete!")


if __name__ == "__main__":
    main()
