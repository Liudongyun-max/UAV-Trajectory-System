#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import cv2
import numpy as np
from typing import Optional
from scipy.signal import savgol_filter

class PixelTrajectoryTracker:
    """二维像素坐标轨迹卡尔曼追踪器，为图像分割提供运动学状态引导"""
    def __init__(self):
        self.state = None  # [x, y, vx, vy]
        self.last_time = None
        self.kf_p = np.eye(4) * 100.0  # 状态协方差
        self.kf_q = np.diag([5.0, 5.0, 10.0, 10.0])  # 过程噪声协方差
        self.kf_r = np.diag([4.0, 4.0])  # 测量噪声协方差
        
    def reset(self):
        self.state = None
        self.last_time = None
        self.kf_p = np.eye(4) * 100.0
        
    def predict(self, curr_time):
        if self.state is None:
            return None
        dt = curr_time - self.last_time if self.last_time is not None else 0.04
        if dt <= 0:
            dt = 0.04
        # 恒速运动模型状态预测
        x, y, vx, vy = self.state
        x_pred = x + vx * dt
        y_pred = y + vy * dt
        self.state = [x_pred, y_pred, vx, vy]
        self.last_time = curr_time
        
        # 协方差预测
        f = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.kf_p = f @ self.kf_p @ f.T + self.kf_q * dt
        return (x_pred, y_pred)
        
    def update(self, meas_x, meas_y, curr_time):
        if self.state is None:
            # 状态初始化，初速度设为 0
            self.state = [meas_x, meas_y, 0.0, 0.0]
            self.last_time = curr_time
            self.kf_p = np.eye(4) * 10.0
            return (meas_x, meas_y)
            
        dt = curr_time - self.last_time if self.last_time is not None else 0.04
        if dt <= 0:
            dt = 0.04
            
        # 观测矩阵 H
        h = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # 计算卡尔曼增益
        s = h @ self.kf_p @ h.T + self.kf_r
        k = self.kf_p @ h.T @ np.linalg.inv(s)
        
        # 状态更新
        z = np.array([meas_x, meas_y])
        y_residual = z - h @ self.state
        self.state = self.state + k @ y_residual
        
        # 协方差更新
        self.kf_p = (np.eye(4) - k @ h) @ self.kf_p
        self.last_time = curr_time
        return (self.state[0], self.state[1])


class Track:
    """
    单个轨迹段对象，维护自身的历史缓存、生命周期、物理平滑状态以及分类预测结果
    """
    def __init__(self, track_id: int, first_pos: tuple[float, float], w: float = 50.0, h: float = 50.0, conf: float = 1.0):
        self.track_id = track_id
        # 历史滑动窗口缓存，保存Dict元素：{'x', 'y', 'px', 'py', 'w', 'h', 'conf'}
        self.history_buffer: list[dict] = []
        self.history_buffer.append({
            'x': first_pos[0], 'y': first_pos[1],
            'px': first_pos[0], 'py': first_pos[1],
            'w': w, 'h': h, 'conf': conf
        })
        self.missing_frames = 0
        self.is_uav = False
        self.classification_prob = None
        self.pred_coords = []
        # 最近锁定的物理位置，用于静止死锁检测
        self.locked_positions: list[tuple[float, float]] = [first_pos]
        self.invalid_dist_frames = 0
        
        # 指数平滑物理防抖变量
        self.smooth_px = first_pos[0]
        self.smooth_py = first_pos[1]
        self.smooth_w = w
        self.smooth_h = h

    def update_smooth_filter(self, px: float, py: float, w: float, h: float, alpha: float = 0.65):
        """
        一阶指数低通滤波： smooth = alpha * val + (1 - alpha) * smooth
        """
        self.smooth_px = alpha * px + (1.0 - alpha) * self.smooth_px
        self.smooth_py = alpha * py + (1.0 - alpha) * self.smooth_py
        self.smooth_w = alpha * w + (1.0 - alpha) * self.smooth_w
        self.smooth_h = alpha * h + (1.0 - alpha) * self.smooth_h


class FeatureTracker:
    """
    多目标智能特征追踪与对齐处理器 (双驱模式，深度移植自 gru detect)
    """
    def __init__(
        self,
        seq_len: int = 20,
        input_dim: int = 12,
        smooth: bool = True,
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.smooth = smooth
        self.normalize = normalize
        
        # 追踪字典：track_id -> Track 实例
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1
        
        # 系统平移对齐偏置量（基于绑定的主轨迹校准）
        self.last_valid_offset = (0.0, 0.0)
        self.offset_initialized = False
        
        # 偏置突变允许的最大跳跃门限
        self.max_offset_jump = 40.0
        # 纯视频模式下，单帧目标中心最大运动位移门限
        self.max_drone_move = 80.0
        # 轨迹无匹配最大容忍帧数，超过该值则删除轨迹
        self.max_missing_frames = 10
        
    def detect_all_centroids(self, frame: np.ndarray, thresh_val: int = 120) -> list[tuple[int, int, float, float, int]]:
        """
        基于绝对灰度值二值化的多目标检测层（含邻近连通域融合机制）
        返回: List of (cX, cY, w, h, max_contrast)
        """
        height, width = frame.shape[:2]
        self.height, self.width = height, width
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 绝对灰度反向二值化处理
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.last_contours = contours
        
        border = 40
        centroids = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            # 放宽面积下限至 2px 以上，以捕捉更远距离下的微弱像素斑点
            if 2 <= area < 20000:
                cX = int(x + w / 2.0)
                cY = int(y + h / 2.0)
                if border <= cX < width - border and border <= cY < height - border:
                    max_contrast = int(120 - np.min(gray[y:y+h, x:x+w]))
                    centroids.append((cX, cY, float(w), float(h), max_contrast))
                    
        # 2. 检测点去重融合机制：物理间距小于 20px 的两个微小域融合成一个
        if len(centroids) > 1:
            merged = []
            used = set()
            for i in range(len(centroids)):
                if i in used:
                    continue
                cX1, cY1, w1, h1, max_contrast1 = centroids[i]
                
                group = [centroids[i]]
                used.add(i)
                for j in range(i + 1, len(centroids)):
                    if j in used:
                        continue
                    cX2, cY2, w2, h2, max_contrast2 = centroids[j]
                    dist = np.hypot(cX1 - cX2, cY1 - cY2)
                    if dist < 20.0:
                        group.append(centroids[j])
                        used.add(j)
                        
                if len(group) == 1:
                    merged.append(centroids[i])
                else:
                    sum_x = sum(g[0] * (g[2] * g[3]) for g in group)
                    sum_y = sum(g[1] * (g[2] * g[3]) for g in group)
                    sum_area = sum(g[2] * g[3] for g in group)
                    
                    merged_cX = int(sum_x / (sum_area + 1e-8))
                    merged_cY = int(sum_y / (sum_area + 1e-8))
                    merged_w = max(g[2] for g in group)
                    merged_h = max(g[3] for g in group)
                    merged_contrast = max(g[4] for g in group)
                    
                    merged.append((merged_cX, merged_cY, merged_w, merged_h, merged_contrast))
            centroids = merged
            
        return centroids
        
    def _find_csv_bound_track_id(self, csv_row: dict) -> Optional[int]:
        if not self.tracks:
            return None
        target_x = csv_row["x_center"] + self.last_valid_offset[0]
        target_y = csv_row["y_center"] + self.last_valid_offset[1]
        best_tid = None
        min_dist = float('inf')
        for tid, track in self.tracks.items():
            last_pt = track.history_buffer[-1]
            dist = np.hypot(last_pt['px'] - target_x, last_pt['py'] - target_y)
            if dist < min_dist:
                min_dist = dist
                best_tid = tid
        return best_tid
        
    def _is_closest_to_csv(self, track: Track, csv_row: dict) -> bool:
        bound_id = self._find_csv_bound_track_id(csv_row)
        return bound_id == track.track_id
        
    def update(self, frame: np.ndarray, csv_row: Optional[dict] = None) -> tuple[Optional[tuple[int, int]], tuple[float, float], bool]:
        height, width = frame.shape[:2]
        self.height, self.width = height, width
        
        scale_area = (width * height) / (1280.0 * 720.0)
        scale_linear = np.sqrt(scale_area)
        current_max_offset_jump = self.max_offset_jump * scale_linear
        current_max_drone_move = self.max_drone_move * scale_linear
        sky_height_limit = int(height * (1100.0 / 1440.0))
        
        # 1. 提取当前帧二值化检测的候选点（调优阈值以优化漏检抑制，提取低对比度目标）
        centroids = self.detect_all_centroids(frame, thresh_val=130)
        if not centroids and not self.tracks:
            centroids = self.detect_all_centroids(frame, thresh_val=140)
            
        # 2. 估计每条轨迹的外推位置用于邻近匹配
        track_predictions = {}
        for tid, track in self.tracks.items():
            last_pt = track.history_buffer[-1]
            pred_x, pred_y = last_pt['px'], last_pt['py']
            if len(track.history_buffer) >= 2:
                prev_pt = track.history_buffer[-2]
                vx = last_pt['px'] - prev_pt['px']
                vy = last_pt['py'] - prev_pt['py']
                pred_x += vx
                pred_y += vy
            track_predictions[tid] = (pred_x, pred_y)
            
        # 3. 贪婪匹配关联
        matched_centroids = set()
        matched_tracks = set()
        match_candidates = []
        for tid, track in self.tracks.items():
            pred_pos = track_predictions[tid]
            for c_idx, c in enumerate(centroids):
                dist = np.hypot(c[0] - pred_pos[0], c[1] - pred_pos[1])
                if dist < current_max_drone_move:
                    match_candidates.append((dist, tid, c_idx))
                    
        match_candidates.sort(key=lambda x: x[0])
        
        for dist, tid, c_idx in match_candidates:
            if tid not in matched_tracks and c_idx not in matched_centroids:
                matched_tracks.add(tid)
                matched_centroids.add(c_idx)
                c = centroids[c_idx]
                track = self.tracks[tid]
                
                # 数据对齐机制 (CSV-Driven)
                if csv_row is not None and self._is_closest_to_csv(track, csv_row):
                    current_x, current_y = csv_row["x_center"], csv_row["y_center"]
                    proposed_offset = (float(c[0] - current_x), float(c[1] - current_y))
                    if not self.offset_initialized:
                        self.last_valid_offset = proposed_offset
                        self.offset_initialized = True
                        aligned_px, aligned_py = c[0], c[1]
                    else:
                        dx = proposed_offset[0] - self.last_valid_offset[0]
                        dy = proposed_offset[1] - self.last_valid_offset[1]
                        offset_dist = np.sqrt(dx**2 + dy**2)
                        if offset_dist < current_max_offset_jump:
                            self.last_valid_offset = proposed_offset
                            aligned_px, aligned_py = c[0], c[1]
                        else:
                            aligned_px = current_x + self.last_valid_offset[0]
                            aligned_py = current_y + self.last_valid_offset[1]
                            
                    track.history_buffer.append({
                        'x': current_x, 'y': current_y,
                        'px': float(aligned_px), 'py': float(aligned_py),
                        'w': float(c[2]), 'h': float(c[3]), 'conf': csv_row.get("detector_confidence", 1.0)
                    })
                else:
                    # 纯视频检测物理更新
                    track.history_buffer.append({
                        'x': float(c[0]), 'y': float(c[1]),
                        'px': float(c[0]), 'py': float(c[1]),
                        'w': float(c[2]), 'h': float(c[3]), 'conf': 1.0
                    })
                    
                track.missing_frames = 0
                track.invalid_dist_frames = 0
                track.locked_positions.append((c[0], c[1]))
                if len(track.locked_positions) > 50:
                    track.locked_positions.pop(0)
                
                # 一阶低通物理平滑防抖
                latest = track.history_buffer[-1]
                track.update_smooth_filter(latest['px'], latest['py'], latest['w'], latest['h'], alpha=0.65)
                
        # 4. 未匹配轨线惯性盲推
        dead_track_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.missing_frames += 1
                if track.missing_frames > self.max_missing_frames:
                    dead_track_ids.append(tid)
                else:
                    # 外推虚假匹配以保持预测链条
                    pred_pos = track_predictions[tid]
                    last_pt = track.history_buffer[-1]
                    track.history_buffer.append({
                        'x': pred_pos[0] - self.last_valid_offset[0],
                        'y': pred_pos[1] - self.last_valid_offset[1],
                        'px': pred_pos[0], 'py': pred_pos[1],
                        'w': last_pt['w'], 'h': last_pt['h'], 'conf': last_pt['conf'] * 0.8
                    })
                    track.invalid_dist_frames += 1
                    track.locked_positions.append(pred_pos)
                    if len(track.locked_positions) > 50:
                        track.locked_positions.pop(0)
                        
                    latest = track.history_buffer[-1]
                    track.update_smooth_filter(latest['px'], latest['py'], latest['w'], latest['h'], alpha=0.65)
                    
        # 注销死亡轨线
        for tid in dead_track_ids:
            del self.tracks[tid]
            
        # 5. 轨迹冲突去重合并机制
        if len(self.tracks) > 1:
            tids = list(self.tracks.keys())
            merged_tids = set()
            for i in range(len(tids)):
                tid1 = tids[i]
                if tid1 in merged_tids or tid1 not in self.tracks:
                    continue
                track1 = self.tracks[tid1]
                last_pt1 = track1.history_buffer[-1]
                
                for j in range(i + 1, len(tids)):
                    tid2 = tids[j]
                    if tid2 in merged_tids or tid2 not in self.tracks:
                        continue
                    track2 = self.tracks[tid2]
                    last_pt2 = track2.history_buffer[-1]
                    
                    dist = np.hypot(last_pt1['px'] - last_pt2['px'], last_pt1['py'] - last_pt2['py'])
                    if dist < 35.0:
                        prob1 = track1.classification_prob if track1.classification_prob is not None else -1.0
                        prob2 = track2.classification_prob if track2.classification_prob is not None else -1.0
                        
                        if abs(prob1 - prob2) < 1e-4:
                            age1 = len(track1.history_buffer)
                            age2 = len(track2.history_buffer)
                            keep_tid = tid1 if age1 >= age2 else tid2
                            delete_tid = tid2 if keep_tid == tid1 else tid1
                        else:
                            keep_tid = tid1 if prob1 >= prob2 else tid2
                            delete_tid = tid2 if keep_tid == tid1 else tid1
                            
                        merged_tids.add(delete_tid)
                        
            for tid in merged_tids:
                if tid in self.tracks:
                    del self.tracks[tid]
                    
        # 6. 统一静态背景死锁阻断
        for tid, track in list(self.tracks.items()):
            is_deadlocked = False
            if len(track.locked_positions) >= 30:
                recent_30 = track.locked_positions[-30:]
                xs = [p[0] for p in recent_30]
                ys = [p[1] for p in recent_30]
                span_x = max(xs) - min(xs)
                span_y = max(ys) - min(ys)
                
                is_near_ground = track.locked_positions[-1][1] >= sky_height_limit
                if is_near_ground and span_x < 5.0 * scale_linear and span_y < 5.0 * scale_linear:
                    is_deadlocked = True
                elif len(track.locked_positions) >= 50:
                    recent_50 = track.locked_positions[-50:]
                    xs_50 = [p[0] for p in recent_50]
                    ys_50 = [p[1] for p in recent_50]
                    span_x_50 = max(xs_50) - min(xs_50)
                    span_y_50 = max(ys_50) - min(ys_50)
                    if span_x_50 < 3.0 * scale_linear and span_y_50 < 3.0 * scale_linear:
                        is_deadlocked = True
                        
            if is_deadlocked:
                del self.tracks[tid]
                
        # 7. 未匹配检测点独立生成新轨迹
        for c_idx, c in enumerate(centroids):
            if c_idx not in matched_centroids:
                if c[1] < sky_height_limit:
                    coords = np.array([[p[0], p[1]] for p in centroids])
                    if len(coords) > 1:
                        dists = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
                        neighbors = np.sum(dists < 150.0 * scale_linear) - 1
                    else:
                        neighbors = 0
                        
                    if neighbors < 4:
                        new_track = Track(
                            track_id=self.next_track_id,
                            first_pos=(float(c[0]), float(c[1])),
                            w=float(c[2]),
                            h=float(c[3]),
                            conf=1.0
                        )
                        if csv_row is not None and self._is_closest_to_csv(new_track, csv_row):
                            current_x, current_y = csv_row["x_center"], csv_row["y_center"]
                            proposed_offset = (float(c[0] - current_x), float(c[1] - current_y))
                            self.last_valid_offset = proposed_offset
                            self.offset_initialized = True
                            new_track.history_buffer[0] = {
                                'x': current_x, 'y': current_y,
                                'px': float(c[0]), 'py': float(c[1]),
                                'w': float(c[2]), 'h': float(c[3]), 'conf': csv_row.get("detector_confidence", 1.0)
                            }
                            new_track.smooth_px = float(c[0])
                            new_track.smooth_py = float(c[1])
                            new_track.smooth_w = float(c[2])
                            new_track.smooth_h = float(c[3])
                            
                        self.tracks[self.next_track_id] = new_track
                        self.next_track_id += 1
                        
        # 8. 限制历史长度为 seq_len
        for track in self.tracks.values():
            if len(track.history_buffer) > self.seq_len:
                track.history_buffer.pop(0)
                
        # 9. 兼容输出最高置信度的主轨迹（严格过滤时序确诊的鸟等背景噪声干扰）
        primary_track = None
        uav_tracks = [t for t in self.tracks.values() if t.is_uav]
        candidate_tracks = [
            t for t in self.tracks.values()
            if t.is_uav or (len(t.history_buffer) < self.seq_len)
        ]
        
        if uav_tracks:
            primary_track = max(uav_tracks, key=lambda t: t.classification_prob)
        elif candidate_tracks:
            primary_track = min(candidate_tracks, key=lambda t: t.track_id)
            
        if primary_track is not None and primary_track.history_buffer:
            drone_pos = (int(primary_track.smooth_px), int(primary_track.smooth_py))
            is_valid = len(primary_track.history_buffer) == self.seq_len
        else:
            drone_pos = None
            is_valid = False
            
        return drone_pos, self.last_valid_offset, is_valid

    def get_track_features(self, track: Track) -> np.ndarray:
        x = np.array([pt['x'] for pt in track.history_buffer], dtype=np.float64)
        y = np.array([pt['y'] for pt in track.history_buffer], dtype=np.float64)
        w = np.array([pt['w'] for pt in track.history_buffer], dtype=np.float64)
        h = np.array([pt['h'] for pt in track.history_buffer], dtype=np.float64)
        conf = np.array([pt['conf'] for pt in track.history_buffer], dtype=np.float64)
        
        if self.smooth and len(x) >= 5:
            x = savgol_filter(x, 5, 2)
            y = savgol_filter(y, 5, 2)
            
        if self.normalize:
            w_ref = getattr(self, "width", 2560.0)
            h_ref = getattr(self, "height", 1440.0)
            x_norm = x / (2.0 * w_ref)
            y_norm = y / (2.0 * h_ref)
            w_norm = w / (2.0 * w_ref)
            h_norm = h / (2.0 * h_ref)
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
