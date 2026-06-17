#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd


class TemporalResampler:
    @staticmethod
    def resample_to_15fps(point_df: pd.DataFrame, start_idx: int, seq_len: int = 25) -> pd.DataFrame | None:
        """
        以 point_df.iloc[start_idx] 为起始帧, 按 15 FPS (时间间隔 1/15s) 向后重采样 seq_len 帧。
        返回重采样后的子 DataFrame。如果点内帧数不够覆盖 15 FPS 约 1.67 秒的时间跨度, 返回 None。
        """
        # 确保 timestamp 正确
        timestamps = point_df["timestamp"].to_numpy(dtype=float)
        t_start = timestamps[start_idx]
        
        # 15 FPS 对应时间跨度 (seq_len - 1) * (1/15)
        # 例如 25 帧对应 24 * 0.066667s = 1.6s
        t_end = t_start + (seq_len - 1) * (1.0 / 15.0)
        
        # 如果 Point 最后一帧的时间戳比 t_end 小, 说明时间范围不够切出完整的 15 FPS 窗口
        if timestamps[-1] < t_end - 1e-4:
            return None
            
        resampled_indices = []
        for k in range(seq_len):
            t_target = t_start + k * (1.0 / 15.0)
            # 寻找距离 t_target 最近的帧索引
            idx = np.argmin(np.abs(timestamps - t_target))
            resampled_indices.append(idx)
            
        sub_df = point_df.iloc[resampled_indices].copy()
        
        # 覆盖时间差列
        sub_df["effective_fps"] = 15.0
        sub_df["window_duration_s"] = float(sub_df["timestamp"].iloc[-1] - sub_df["timestamp"].iloc[0])
        # 重采样后的帧差
        sub_df["frame_delta_t_s"] = sub_df["timestamp"].diff().fillna(1.0 / 15.0)
        sub_df["fps_profile"] = 15
        
        return sub_df
