#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from uav_distance_pipeline.config import ROOT


def test_split_leakage():
    audit_path = ROOT / "reports" / "run017_windows_v2_audited" / "split_leakage_audit.json"
    if not audit_path.exists():
        pytest.skip("split_leakage_audit.json not found, skip test")
        
    with audit_path.open("r", encoding="utf-8") as f:
        audit = json.load(f)
        
    assert audit.get("point_overlap_count", 999) == 0, f"Point leakage detected: {audit.get('split_point_overlap')}"
    assert audit.get("session_overlap_count", 999) == 0, f"Session leakage detected: {audit.get('split_session_overlap')}"
