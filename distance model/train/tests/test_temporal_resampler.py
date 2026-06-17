#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from training.data.temporal_resampler import TemporalResampler


def test_temporal_resampler_logic():
    # 模拟一个含有 100 帧 25 FPS (每帧差 0.04s) 的 Point df
    timestamps = np.linspace(0.0, 99 * 0.04, 100) # 0.0s 到 3.96s
    df = pd.DataFrame({
        "timestamp": timestamps,
        "true_range_m": np.random.randn(100),
        "is_missing": np.zeros(100),
        "is_interpolated": np.zeros(100)
    })
    
    # 从第 0 帧开始重采样成 15 FPS 的 25 帧序列 (时间段: 0.0 到 24 * (1/15) = 1.6s)
    sub_df = TemporalResampler.resample_to_15fps(df, start_idx=0, seq_len=25)
    
    assert sub_df is not None
    assert len(sub_df) == 25
    assert sub_df["effective_fps"].iloc[0] == 15.0
    
    # 验证重采样帧时间戳的邻近性 (1.6s 对应的真实时间戳应非常接近其 k*(1/15))
    expected_times = [k * (1.0 / 15.0) for k in range(25)]
    actual_times = sub_df["timestamp"].to_numpy()
    
    for exp, act in zip(expected_times, actual_times):
        # 25 FPS 数据每帧精度为 0.04s, 邻近误差应小于 0.02s (一半的分辨率)
        assert abs(exp - act) <= 0.021, f"Timestamp matching error too large: expected={exp}, actual={act}"
