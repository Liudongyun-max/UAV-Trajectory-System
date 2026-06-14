"""
YOLO 适配版时序特征追踪器 (tracker.py)
"""
import numpy as np
import cv2
from scipy.signal import savgol_filter
from typing import Tuple, List, Optional, Dict

class FeatureTracker:
    def __init__(
        self,
        seq_len: int = 20,
        input_dim: int = 8,
        smooth: bool = True,
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.smooth = smooth
        self.normalize = normalize
        
        self.history_buffer: List[Dict] = []
        self.invalid_dist_frames = 0
        self.locked_positions: List[Tuple[int, int]] = []
        
        self.max_drone_move = 130.0 # 放宽单帧最大位移匹配限额以支持高速机动 (130px)

    def update(self, frame: np.ndarray, detections: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Tuple[float, float], bool]:
        """
        利用 YOLO 检测框更新追踪滑动窗口
        返回:
            drone_pos: 无人机当前的实际显示中心像素 (x, y)，丢失时为 None
            offset: 校正偏移量 (此处固定为 0,0，因为是纯视频模式)
            is_valid: 历史缓存是否填满以供 GRU 预测
        """
        height, width = frame.shape[:2]
        
        # 将 detections [N, 6] (x1, y1, x2, y2, conf, cls) 转换为中心点和大小
        candidates = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # 过滤类别: 只保留 class_id = 0 (通常是 drone)
            if int(cls) == 0:
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                w = float(x2 - x1)
                h = float(y2 - y1)
                candidates.append((cX, cY, w, h, conf))
                
        drone_pos = None
        drone_w, drone_h = 50.0, 50.0
        drone_conf = 1.0
        
        last_pos = None
        if len(self.history_buffer) > 0:
            last_pos = (int(self.history_buffer[-1]['px']), int(self.history_buffer[-1]['py']))
            
        if last_pos is not None:
            # 1. 追踪模式：在 60px 半径内匹配最近的检测框
            best_candidate = None
            min_dist = float('inf')
            for c in candidates:
                dist = np.hypot(c[0] - last_pos[0], c[1] - last_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_candidate = c
                    
            if min_dist < self.max_drone_move and best_candidate is not None:
                drone_pos = (best_candidate[0], best_candidate[1])
                drone_w, drone_h = best_candidate[2], best_candidate[3]
                drone_conf = float(best_candidate[4])
                self.invalid_dist_frames = 0
            else:
                # 局域未匹配到目标，维持上一帧位置，并判定是否死锁
                drone_pos = last_pos
                drone_w = self.history_buffer[-1]['w']
                drone_h = self.history_buffer[-1]['h']
                drone_conf = self.history_buffer[-1]['conf']
                self.invalid_dist_frames += 1
        else:
            # 2. 初始捕获模式
            if candidates:
                best = max(candidates, key=lambda x: x[4])
                drone_pos = (best[0], best[1])
                drone_w, drone_h = best[2], best[3]
                drone_conf = float(best[4])
                
        # 3. 统一死锁与静态背景粘滞判定机制
        is_deadlocked = False
        if drone_pos is not None:
            temp_locked = list(self.locked_positions) + [drone_pos]
            
            # 机制 A: 基于最近 20 帧坐标标准差的硬核静态背景粘滞死锁判定
            # 无人机即使悬停在风中也会有一定像素抖动，如果 20 帧内标准差小于 1.2px，绝对是建筑物纹理死锁
            if len(self.history_buffer) >= 20:
                xs = [pt['x'] for pt in self.history_buffer]
                ys = [pt['y'] for pt in self.history_buffer]
                if np.std(xs) < 1.2 and np.std(ys) < 1.2:
                    is_deadlocked = True
                    print(f"--> [FeatureTracker] Deadlock standard deviation limit reached (std_x={np.std(xs):.2f}, std_y={np.std(ys):.2f})!")

            # 机制 B: 如果连续 12 帧局域未匹配到 YOLO 真实目标，加速全局捕获唤醒
            if self.invalid_dist_frames >= 12:
                is_deadlocked = True
                print(f"--> [FeatureTracker] Target lost for {self.invalid_dist_frames} frames. Triggering fast capture reset.")

            if not is_deadlocked and len(temp_locked) >= 30:
                xs = [p[0] for p in temp_locked if p is not None]
                ys = [p[1] for p in temp_locked if p is not None]
                if xs and ys:
                    span_x = max(xs) - min(xs)
                    span_y = max(ys) - min(ys)
                    if span_x < 5 and span_y < 5:
                        is_deadlocked = True
                    elif len(temp_locked) >= 50 and span_x < 8 and span_y < 8 and self.invalid_dist_frames >= 20:
                        is_deadlocked = True

        if is_deadlocked:
            print(f"--> [FeatureTracker] Static deadlock detected! Resetting tracker. locked_positions len: {len(self.locked_positions)}")
            self.history_buffer.clear()
            self.locked_positions.clear()
            self.invalid_dist_frames = 0
            drone_pos = None

        if drone_pos is not None:
            self.locked_positions.append(drone_pos)
            if len(self.locked_positions) > 50:
                self.locked_positions.pop(0)
                
            self.history_buffer.append({
                'x': float(drone_pos[0]), 'y': float(drone_pos[1]),
                'px': float(drone_pos[0]), 'py': float(drone_pos[1]),
                'w': drone_w, 'h': drone_h, 'conf': drone_conf
            })
            
        if len(self.history_buffer) > self.seq_len:
            self.history_buffer.pop(0)
            
        is_valid = len(self.history_buffer) == self.seq_len
        return drone_pos, (0.0, 0.0), is_valid

    def get_features(self) -> np.ndarray:
        """
        提取用于 GRU 预测的时序归一化特征
        """
        x = np.array([pt['x'] for pt in self.history_buffer], dtype=np.float64)
        y = np.array([pt['y'] for pt in self.history_buffer], dtype=np.float64)
        w = np.array([pt['w'] for pt in self.history_buffer], dtype=np.float64)
        h = np.array([pt['h'] for pt in self.history_buffer], dtype=np.float64)
        conf = np.array([pt['conf'] for pt in self.history_buffer], dtype=np.float64)
        
        if self.smooth and len(x) >= 5:
            x = savgol_filter(x, 5, 2)
            y = savgol_filter(y, 5, 2)
            
        if self.normalize:
            x_norm = x / 2560.0
            y_norm = y / 1440.0
            w_norm = w / 2560.0
            h_norm = h / 1440.0
        else:
            x_norm = x
            y_norm = y
            w_norm = w
            h_norm = h
            
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
        
        if self.input_dim == 12:
            aspect_ratio = w_norm / (h_norm + 1e-8)
            feature_list.extend([w_norm, h_norm, aspect_ratio, conf])
            
        features = np.stack(feature_list, axis=-1)
        return features.astype(np.float32)
