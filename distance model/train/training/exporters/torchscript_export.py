"""TorchScript导出"""
import torch
import torch.nn as nn
from pathlib import Path


def export_to_torchscript(
    model: nn.Module,
    save_path: str,
    input_dim: int = 8,
    seq_len: int = 20,
    method: str = "trace",
):
    """
    导出模型为TorchScript格式
    
    Args:
        model: PyTorch模型
        save_path: 保存路径
        input_dim: 输入特征维度
        seq_len: 序列长度
        method: 导出方法 (trace/script)
    """
    model.eval()
    
    dummy_input = torch.randn(1, seq_len, input_dim)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if method == "trace":
        scripted_model = torch.jit.trace(model, dummy_input)
    elif method == "script":
        scripted_model = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    scripted_model.save(str(save_path))
    
    print(f"Model exported to {save_path}")
