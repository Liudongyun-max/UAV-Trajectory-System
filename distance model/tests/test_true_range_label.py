#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from uav_distance_pipeline.config import ROOT


def test_true_range_label_logic():
    dataset_path = ROOT / "dataset" / "run017_windows_v2_audited" / "frame_dataset.csv"
    if not dataset_path.exists():
        pytest.skip("frame_dataset.csv not found, skip test")
        
    df = pd.read_csv(dataset_path)
    
    # 抽样几行验证: true_range_m - fused_range_m 是否等于 residual_range_m
    sample = df.sample(min(100, len(df)), random_state=42)
    for _, row in sample.iterrows():
        true_r = float(row["true_range_m"])
        fused_r = float(row["fused_range_m"])
        res_r = float(row["residual_range_m"])
        # residual_range_m 在写 CSV 时保留了 9 位小数，我们做 epsilon 的比对
        assert abs(true_r - fused_r - res_r) < 1e-6, f"True range residual math mismatch! true_range: {true_r}, fused: {fused_r}, residual: {res_r}"
