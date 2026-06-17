#!/usr/bin/env encoding
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from uav_distance_pipeline.config import ROOT


def test_gru_export_contract():
    export_dir = ROOT / "exports" / "gru_input" / "run017"
    if not export_dir.exists():
        pytest.skip("Export directory not found, skip test")
        
    # 1. 检查必要文件存在
    required_files = [
        "frame_predictions.csv", "point_manifest.csv", "session_manifest.csv",
        "fold_assignments.csv", "feature_schema.json", "normalization.json", "export_manifest.json"
    ]
    for rf in required_files:
        assert (export_dir / rf).exists(), f"Missing required export file: {rf}"
        
    # 2. 验证 feature_schema.json 内容
    with (export_dir / "feature_schema.json").open("r", encoding="utf-8") as f:
        schema = json.load(f)
        
    features = schema.get("feature_columns", [])
    
    # scene_profile, nominal_distance, distance_bin 不应该在特征列表里
    banned = ["scene_profile", "nominal_distance", "distance_bin"]
    for b in banned:
        for f_name in features:
            assert b not in f_name.lower(), f"Banned column {b} leaked to GRU feature columns: {f_name}"
            
    # 3. 验证 frame_predictions 中的 is_missing, is_interpolated, frame_delta_t_s 列是否存在且为数值
    df_preds = pd.read_csv(export_dir / "frame_predictions.csv")
    assert "is_missing" in df_preds.columns, "is_missing column missing in export"
    assert "is_interpolated" in df_preds.columns, "is_interpolated column missing in export"
    assert "frame_delta_t_s" in df_preds.columns, "frame_delta_t_s column missing in export"
    
    assert not df_preds["is_missing"].isnull().any(), "is_missing contains NaN values"
    assert not df_preds["is_interpolated"].isnull().any(), "is_interpolated contains NaN values"
    assert not df_preds["frame_delta_t_s"].isnull().any(), "frame_delta_t_s contains NaN values"
