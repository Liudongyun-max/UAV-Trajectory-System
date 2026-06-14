"""ONNX导出"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_dim: int = 8,
    seq_len: int = 20,
    opset_version: int = 11,
    dynamic_axes: Optional[dict] = None,
):
    """
    导出模型为ONNX格式
    
    Args:
        model: PyTorch模型
        save_path: 保存路径
        input_dim: 输入特征维度
        seq_len: 序列长度
        opset_version: ONNX opset版本
        dynamic_axes: 动态轴配置
    """
    model.eval()
    
    dummy_input = torch.randn(1, seq_len, input_dim)
    
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "future_offsets": {0: "batch_size"},
        }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["logits", "future_offsets"],
        dynamic_axes=dynamic_axes,
    )
    
    print(f"Model exported to {save_path}")
    
    try:
        import onnx
        onnx_model = onnx.load(str(save_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")
    except ImportError:
        print("onnx package not installed, skipping validation")
