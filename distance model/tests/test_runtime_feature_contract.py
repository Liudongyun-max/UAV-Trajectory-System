#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import hashlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_feature_schema_hash_contract():
    # 1. 验证当前的 feature_schema.json 和 runtime_feature_contract.json 是匹配的
    schema_path = ROOT / "models" / "feature_schema.json"
    contract_path = ROOT / "reports" / "runtime_distance_stability" / "runtime_feature_contract.json"
    
    assert schema_path.exists(), "models/feature_schema.json does not exist"
    assert contract_path.exists(), "runtime_feature_contract.json does not exist"
    
    with open(schema_path, "r", encoding="utf-8") as sf:
        schema_data = json.load(sf)
        
    with open(contract_path, "r", encoding="utf-8") as cf:
        contract_data = json.load(cf)
        
    normalized_schema_str = json.dumps(schema_data, sort_keys=True)
    runtime_hash = hashlib.sha256(normalized_schema_str.encode("utf-8")).hexdigest()
    
    expected_hash = contract_data.get("model_feature_schema_hash", "")
    
    assert runtime_hash == expected_hash, f"Hash mismatch: calculated {runtime_hash}, expected {expected_hash}"
    assert len(schema_data.get("feature_columns", [])) == 16, "Feature columns must be exactly 16"
    assert schema_data.get("input_dim") == 16, "input_dim must be 16"

def test_feature_schema_tamper_detection():
    # 2. 模拟篡改特征定义，验证计算的哈希会发生变化，从而能够被系统拦截
    schema_tampered = {
        "feature_columns": [
            "bbox_width_px",
            "bbox_height_px"
        ],
        "input_dim": 2
    }
    
    contract_path = ROOT / "reports" / "runtime_distance_stability" / "runtime_feature_contract.json"
    with open(contract_path, "r", encoding="utf-8") as cf:
        contract_data = json.load(cf)
    expected_hash = contract_data.get("model_feature_schema_hash", "")
    
    normalized_schema_str = json.dumps(schema_tampered, sort_keys=True)
    tampered_hash = hashlib.sha256(normalized_schema_str.encode("utf-8")).hexdigest()
    
    assert tampered_hash != expected_hash, "Tampered schema hash should not match expected contract hash"
