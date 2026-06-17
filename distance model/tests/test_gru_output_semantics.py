#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from pathlib import Path
import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def test_gru_model_inference_contract():
    sys.path.insert(0, str(ROOT / "train"))
    from training.models.gru_distance_stabilizer import GRUDistanceStabilizer
    
    schema_path = ROOT / "models" / "feature_schema.json"
    assert schema_path.exists()
    
    with open(schema_path, "r", encoding="utf-8") as sf:
        schema = json.load(sf)
    
    input_dim = schema.get("input_dim", 16)
    seq_len = schema.get("sequence_length", 25)
    
    model = GRUDistanceStabilizer(input_dim=input_dim)
    dummy_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        output, _ = model(dummy_input)
        
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"

def test_gru_load_weights_and_run():
    sys.path.insert(0, str(ROOT / "train"))
    from training.models.gru_distance_stabilizer import GRUDistanceStabilizer
    
    model_path = ROOT / "models" / "best.pth"
    schema_path = ROOT / "models" / "feature_schema.json"
    
    if not model_path.exists():
        pytest.skip("models/best.pth not found, skip load weights test")
        
    with open(schema_path, "r", encoding="utf-8") as sf:
        schema = json.load(sf)
        
    input_dim = schema.get("input_dim", 16)
    
    model = GRUDistanceStabilizer(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    dummy_input = torch.zeros(1, 25, input_dim, dtype=torch.float32)
    with torch.no_grad():
        output, _ = model(dummy_input)
        
    val = output.item()
    assert isinstance(val, float)
