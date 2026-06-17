"""模型注册表"""
from typing import Dict, Type
import torch.nn as nn


class ModelRegistry:
    """
    模型注册表
    
    用于管理不同类型的轨迹识别模型
    """
    
    _models: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        注册模型装饰器
        
        Usage:
            @ModelRegistry.register("gru")
            class GRUModel(nn.Module):
                ...
        """
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """获取已注册的模型"""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """创建模型实例"""
        model_class = cls.get(name)
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有已注册的模型"""
        return list(cls._models.keys())
