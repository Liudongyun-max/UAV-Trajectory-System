"""
YOLO + GRU 端到端级联视频处理管道 (pipeline.py)
"""
import os
import cv2
import torch
import numpy as np
from typing import Optional
from pathlib import Path
from detector import get_detector, MockFallbackImpl
from tracker import FeatureTracker
from model import UAVTrajectoryNet

class YoloGRUPipeline:
    def __init__(
        self,
        video_path: str,
        yolo_model_path: Optional[str] = None,
        gru_model_path: str = "F:/UAV Trajectory System/gru detect/models/gru_baseline.pth",
        device: str = "auto",
        conf_thres: float = 0.3,
        nms_thres: float = 0.45
    ):
        self.video_path = video_path
        self.yolo_model_path = yolo_model_path
        self.gru_model_path = gru_model_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        
        # 决定设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"--> [Pipeline] Using device: {self.device}")
        
        # 1. 初始化检测器
        self.detector = get_detector(self.yolo_model_path, self.conf_thres, self.nms_thres, device=self.device)
        self.fallback_detector = MockFallbackImpl(self.conf_thres, self.nms_thres)
        
        # 2. 初始化 GRU 轨迹模型
        self.gru_model = UAVTrajectoryNet.load_from_checkpoint(self.gru_model_path, device=self.device)
        self.gru_model.eval()
        
        # 3. 初始化追踪器 (使用模型实际输入维度进行自适应配置)
        self.tracker = FeatureTracker(seq_len=20, input_dim=self.gru_model.input_dim, smooth=True, normalize=True)
        
        # 4. 打开视频
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
            
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not np.isfinite(self.fps) or self.fps <= 1e-3:
            self.fps = 20.0
            
        print(f"--> [Pipeline] Video specs: {self.width}x{self.height} @ {self.fps:.2f} FPS, total {self.total_frames} frames")

    def run(self, output_path: str, max_frames: int = -1):
        """
        运行完整的端到端视频检测预测，并保存为输出视频
        """
        # 创建输出文件夹
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 打开视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")
            
        print(f"--> [Pipeline] Processing video. Saving output to: {output_path}")
        
        frame_idx = 0
        while True:
            if max_frames > 0 and frame_idx >= max_frames:
                print(f"--> [Pipeline] Reached max_frames limit: {max_frames}. Terminating early.")
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break
                
            rendered, drone_pos, is_valid, _ = self.process_single_frame(
                frame, frame_idx, show_box=True, show_history=True, show_pred_red_blue=False
            )
            
            # 写入输出视频
            writer.write(rendered)
            
            if frame_idx % 100 == 0:
                pos_str = f"({drone_pos[0]}, {drone_pos[1]})" if drone_pos else "None"
                print(f"--> [Pipeline] Frame {frame_idx}/{self.total_frames} | Pos: {pos_str} | Valid: {is_valid}")
                
            frame_idx += 1
            
        self.cap.release()
        writer.release()
        print(f"--> [Pipeline] Done. Output video saved successfully to: {output_path}")

    def process_single_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        show_box: bool = True,
        show_history: bool = True,
        show_pred_red_blue: bool = False
    ):
        """
        处理单帧并返回渲染图像，以及追踪相关状态
        """
        # A. YOLO检测 (时空级联无损 ROI 推理与全局孤立候选细筛)
        last_pos = None
        if len(self.tracker.history_buffer) > 0:
            last_pos = (int(self.tracker.history_buffer[-1]['px']), int(self.tracker.history_buffer[-1]['py']))
            
        dets = []
        if last_pos is not None:
            # 1. 局部无损 ROI 追踪模式：以 last_pos 为中心裁剪 640x640 图像进行精细识别
            cX, cY = last_pos
            x1 = max(0, cX - 320)
            y1 = max(0, cY - 320)
            if x1 + 640 > self.width:
                x1 = self.width - 640
            if y1 + 640 > self.height:
                y1 = self.height - 640
            x1 = max(0, x1)
            y1 = max(0, y1)
            
            x2 = min(self.width, x1 + 640)
            y2 = min(self.height, y1 + 640)
            
            roi_img = frame[y1:y2, x1:x2]
            roi_dets = self.detector.detect(roi_img)
            
            for det in roi_dets:
                rx1, ry1, rx2, ry2, conf, cls = det
                if int(cls) == 0:
                    print(f"--> [Pipeline] [Tracker ROI] Found UAV at absolute ({(rx1+rx2)/2.0 + x1:.1f}, {(ry1+ry2)/2.0 + y1:.1f}) with Conf: {conf:.4f}")
                dets.append([rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1, conf, cls])
        else:
            # 2. 全局捕获模式：双轨级联捕获机制
            # 轨 A：优先在全图尺度运行 YOLO 检测以实现近/中程目标的直接捕获，避免在大楼密集噪点中被 Mock 孤立度算法误杀
            global_yolo_dets = self.detector.detect(frame)
            for det in global_yolo_dets:
                gx1, gy1, gx2, gy2, gconf, gcls = det
                if int(gcls) == 0:
                    print(f"--> [Pipeline] [Global YOLO Capture] Found UAV at absolute ({(gx1+gx2)/2.0:.1f}, {(gy1+gy2)/2.0:.1f}) with Conf: {gconf:.4f}")
                    dets.append([gx1, gy1, gx2, gy2, gconf, gcls])
            
            # 轨 B：若全局未检出（可能由于超远距离微小目标退化），启动 Mock 亮点定位 + 256x256 局部无损细筛进行高分辨率兜底
            if len(dets) == 0:
                fallback_dets = self.fallback_detector.detect(frame)
                for f_det in fallback_dets[:3]:
                    fx1, fy1, fx2, fy2, f_conf, f_cls = f_det
                    fcX = int((fx1 + fx2) / 2.0)
                    fcY = int((fy1 + fy2) / 2.0)
                    
                    rx1 = max(0, fcX - 128)
                    ry1 = max(0, fcY - 128)
                    if rx1 + 256 > self.width:
                        rx1 = self.width - 256
                    if ry1 + 256 > self.height:
                        ry1 = self.height - 256
                    rx1 = max(0, rx1)
                    ry1 = max(0, ry1)
                    
                    rx2 = min(self.width, rx1 + 256)
                    ry2 = min(self.height, ry1 + 256)
                    
                    roi_img = frame[ry1:ry2, rx1:rx2]
                    yolo_roi_dets = self.detector.detect(roi_img)
                    for y_det in yolo_roi_dets:
                        yx1, yy1, yx2, yy2, yconf, ycls = y_det
                        if int(ycls) == 0:
                            print(f"--> [Pipeline] [ROI Fallback Capture] Found UAV at absolute ({(yx1+yx2)/2.0 + rx1:.1f}, {(yy1+yy2)/2.0 + ry1:.1f}) with Conf: {yconf:.4f}")
                            dets.append([yx1 + rx1, yy1 + ry1, yx2 + rx1, yy2 + ry1, yconf, ycls])
                            break
                        
        dets = np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)
        
        # B. 追踪关联
        drone_pos, _, is_valid = self.tracker.update(frame, dets)
        
        # 创建渲染图
        rendered = frame.copy()
        
        # C. 绘制历史轨迹 (纯绿色，高能见度)
        if show_history and len(self.tracker.history_buffer) > 1:
            for i in range(1, len(self.tracker.history_buffer)):
                p0 = self.tracker.history_buffer[i - 1]
                p1 = self.tracker.history_buffer[i]
                pt0 = (int(p0["px"]), int(p0["py"]))
                pt1 = (int(p1["px"]), int(p1["py"]))
                cv2.line(rendered, pt0, pt1, (0, 255, 0), 2, cv2.LINE_AA)
                
        # D. 绘制当前黄色目标框
        if show_box and drone_pos is not None:
            latest = self.tracker.history_buffer[-1]
            w_box = max(30, int(latest.get("w", 50)))
            h_box = max(30, int(latest.get("h", 50)))
            x1 = drone_pos[0] - w_box // 2
            y1 = drone_pos[1] - h_box // 2
            x2 = drone_pos[0] + w_box // 2
            y2 = drone_pos[1] + h_box // 2
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
            # 绘制置信度文本
            cv2.putText(rendered, f"UAV:{latest.get('conf', 1.0):.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        
        # E. GRU 推理与绘制未来轨迹
        classification_prob = None
        if is_valid:
            feats_np = self.tracker.get_features()
            feats_t = torch.tensor(feats_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits, pred_offsets = self.gru_model(feats_t)
                pred_offsets = pred_offsets.squeeze(0).cpu().numpy()
                classification_prob = torch.sigmoid(logits).item()
                
            if drone_pos is not None:
                current_x, current_y = drone_pos
                scale_x = self.width / 2.0
                scale_y = self.height / 2.0
                
                pred_coords = []
                for off in pred_offsets:
                    px = int(current_x + off[0] * scale_x)
                    py = int(current_y + off[1] * scale_y)
                    pred_coords.append((px, py))
                    
                # 绘制预测航迹：根据 show_pred_red_blue 选择红线蓝点或原本的青色
                line_color = (0, 0, 255) if show_pred_red_blue else (255, 191, 0)
                point_color = (255, 0, 0) if show_pred_red_blue else (255, 191, 0)
                
                for i in range(1, len(pred_coords)):
                    cv2.line(rendered, pred_coords[i-1], pred_coords[i], line_color, 2, cv2.LINE_AA)
                for coord in pred_coords:
                    cv2.circle(rendered, coord, 5, point_color, -1, cv2.LINE_AA)
                    
        # F. 绘制 HUD 信息板
        self._draw_hud(rendered, frame_idx, is_valid, classification_prob)
        
        return rendered, drone_pos, is_valid, classification_prob

    def _draw_hud(self, frame: np.ndarray, frame_idx: int, is_valid: bool, classification_prob: Optional[float]):
        """
        绘制左上角的透明状态面板
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 16), (460, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Frame: {frame_idx}/{self.total_frames}", (30, 48), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"System: YOLOv5 + GRU Cascade Tracker", (30, 80), font, 0.52, (255, 255, 0), 1, cv2.LINE_AA)
        
        if classification_prob is None:
            state_text = "WAITING TARGET/HISTORY" if not is_valid else "PREDICTING"
            cv2.putText(frame, f"State: {state_text}", (30, 112), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            is_noise = classification_prob < 0.5
            cls_text = "ANOMALY/NOISE" if is_noise else "TRUE UAV"
            cls_color = (0, 0, 255) if is_noise else (0, 255, 0)
            cv2.putText(frame, "State: ", (30, 112), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{cls_text} ({classification_prob:.4f})", (90, 112), font, 0.55, cls_color, 1, cv2.LINE_AA)
