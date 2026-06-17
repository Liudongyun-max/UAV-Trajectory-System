#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from uav_distance_pipeline.config import ROOT


def test_feature_columns_no_leakage():
    schema_path = ROOT / "dataset" / "run017_windows_v2_audited" / "feature_schema.json"
    if not schema_path.exists():
        pytest.skip("feature_schema.json not found, skip test")
        
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
        
    features = schema.get("feature_columns", [])
    
    banned = ["true_range", "residual_range", "residual", "nominal_distance", "distance_bin"]
    for feat in features:
        feat_lower = feat.lower()
        for b in banned:
            assert b not in feat_lower, f"Feature leakage detected! Feature '{feat}' contains banned keyword '{b}'"
