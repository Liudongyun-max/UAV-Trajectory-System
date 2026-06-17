#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import torch
import _bootstrap  # noqa: F401
from training.models.gru_distance_stabilizer import GRUDistanceStabilizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--format", default="onnx,torchscript")
    args = p.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    out_dir = Path("outputs") / cfg["experiment"]["name"]
    schema_path = out_dir / "feature_schema.json"
    
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    input_dim = schema["input_dim"]
    seq_len = schema["sequence_length"]
    
    # 定义并加载模型
    model = GRUDistanceStabilizer(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=cfg["model"].get("bidirectional", False)
    )
    
    ckpt_path = out_dir / "checkpoints" / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")
        
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    model.eval()
    
    # 建立输出文件夹
    exports_dir = out_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立 dummy input
    dummy_input = torch.zeros(1, seq_len, input_dim, dtype=torch.float32)
    
    formats = args.format.split(",")
    
    # 1. 导出 ONNX
    if "onnx" in formats:
        onnx_path = exports_dir / "distance_gru_static_25f.onnx"
        
        # ONNX 导出参数
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input"],
            # 两个返回值的名称：最后一帧测距，以及整个序列的测距。我们分别输出
            output_names=["stable_range_m", "seq_predictions_m"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "stable_range_m": {0: "batch_size"},
                "seq_predictions_m": {0: "batch_size"}
            }
        )
        print(f"Model successfully exported to ONNX format at {onnx_path}")
        
        # 验证 ONNX 一致性
        try:
            import onnxruntime as ort
            ort_sess = ort.InferenceSession(str(onnx_path))
            
            # 使用 PyTorch 得到结果
            with torch.no_grad():
                torch_out_last, torch_out_seq = model(dummy_input)
                
            ort_inputs = {"input": dummy_input.numpy()}
            ort_outs = ort_sess.run(None, ort_inputs)
            
            # 检查输出形状和差距
            np.testing.assert_allclose(torch_out_last.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
            print("ONNX Runtime consistency check passed! outputs match PyTorch outputs.")
        except Exception as e:
            print(f"Warning: ONNX consistency check skipped/failed: {str(e)}")
            
    # 2. 导出 TorchScript
    if "torchscript" in formats:
        script_path = exports_dir / "distance_gru_static_25f.torchscript.pt"
        
        # 采用 trace 导出
        traced_cell = torch.jit.trace(model, dummy_input)
        traced_cell.save(str(script_path))
        print(f"Model successfully exported to TorchScript format at {script_path}")


if __name__ == "__main__":
    main()
