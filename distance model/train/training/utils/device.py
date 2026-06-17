"""设备管理"""
import torch


def get_device(device: str = "auto") -> torch.device:
    """
    获取计算设备
    
    Args:
        device: 设备指定 (auto/cuda/cpu)
    
    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


def get_device_info() -> dict:
    """获取设备信息"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return info
