#!/usr/bin/env encoding
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from training.models.gru_distance_stabilizer import GRUDistanceStabilizer


def test_onnx_and_torchscript_export_consistency():
    out_dir = Path("outputs") / "run017_distance_gru_static_25f"
    exports_dir = out_dir / "exports"
    
    if not (exports_dir / "distance_gru_static_25f.onnx").exists():
        pytest.skip("ONNX model not exported yet, skip export consistency test")
        
    onnx_path = exports_dir / "distance_gru_static_25f.onnx"
    ts_path = exports_dir / "distance_gru_static_25f.torchscript.pt"
    
    schema_path = out_dir / "feature_schema.json"
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    input_dim = schema["input_dim"]
    
    # 载入 PyTorch 模型
    model = GRUDistanceStabilizer(input_dim=input_dim)
    model.load_state_dict(torch.load(out_dir / "checkpoints" / "best.pth", map_location="cpu"))
    model.eval()
    
    dummy = torch.randn(1, 25, input_dim)
    
    # 1. 验证 JIT TorchScript 一致性
    ts_model = torch.jit.load(str(ts_path))
    ts_model.eval()
    with torch.no_grad():
        py_last, py_seq = model(dummy)
        ts_last, ts_seq = ts_model(dummy)
        
    np.testing.assert_allclose(py_last.numpy(), ts_last.numpy(), rtol=1e-4, atol=1e-5)
    
    # 2. 验证 ONNX runtime 一代一致性
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path))
        ort_outs = sess.run(None, {"input": dummy.numpy()})
        np.testing.assert_allclose(py_last.numpy(), ort_outs[0], rtol=1e-3, atol=1e-5)
    except ImportError:
        pass # 环境可能没有 onnxruntime
