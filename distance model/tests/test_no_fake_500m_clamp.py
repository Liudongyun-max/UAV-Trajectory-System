#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def test_no_500m_clamping_in_code():
    gui_file = ROOT / "distance_model_gui.py"
    assert gui_file.exists()
    
    with open(gui_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 查找是否有类似 distance = min(..., 500) 或 np.clip(..., 0, 500) 的代码
    # 匹配诸如: min(..., 500.0), np.clip(..., 500), np.clip(..., 500.0)
    clamp_patterns = [
        r"min\([^,]+,\s*500(?:\.0)?\)",
        r"np\.clip\([^,]+,\s*[^,]+,\s*500(?:\.0)?\)",
        r"clip\([^,]+,\s*[^,]+,\s*500(?:\.0)?\)"
    ]
    
    for pattern in clamp_patterns:
        matches = re.findall(pattern, content)
        assert len(matches) == 0, f"Detected potential 500m hard-clamp logic: {matches}"

def test_invalid_measurement_gives_nan():
    import sys
    sys.path.insert(0, str(ROOT))
    from distance_model_gui import DistanceMeasurement
    
    m = DistanceMeasurement(
        raw_unclipped_range_m=float('nan'),
        robust_fused_range_m=float('nan'),
        measurement_valid=False,
        measurement_quality=0.0,
        invalid_reason="TARGET_MISSING"
    )
    
    assert np.isnan(m.raw_unclipped_range_m) or m.raw_unclipped_range_m is None
    assert np.isnan(m.robust_fused_range_m) or m.robust_fused_range_m is None
    assert m.measurement_valid is False
