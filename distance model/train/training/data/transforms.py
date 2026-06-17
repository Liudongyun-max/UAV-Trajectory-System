"""数据变换"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class FeatureTransform:
    """
    特征变换
    
    支持:
    - Savitzky-Golay平滑
    - 高斯噪声增强
    - 随机遮挡
    """
    
    def __init__(
        self,
        smooth: bool = False,
        smooth_window: int = 5,
        smooth_polyorder: int = 2,
        add_noise: bool = False,
        noise_std: float = 0.001,
        random_occlusion: bool = False,
        occlusion_prob: float = 0.1,
    ):
        self.smooth = smooth
        self.smooth_window = smooth_window
        self.smooth_polyorder = smooth_polyorder
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_occlusion = random_occlusion
        self.occlusion_prob = occlusion_prob
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        应用变换
        
        Args:
            features: [seq_len, 8] 特征张量
        
        Returns:
            变换后的特征
        """
        if self.smooth:
            features = self._savitzky_golay_smooth(features)
        
        if self.add_noise:
            features = self._add_gaussian_noise(features)
        
        if self.random_occlusion:
            features = self._random_occlusion(features)
        
        return features
    
    def _savitzky_golay_smooth(self, features: torch.Tensor) -> torch.Tensor:
        """Savitzky-Golay平滑"""
        # 注意：平滑工作已在 dataset.py 解算一阶和二阶差分前的前置阶段完成，
        # 此处直接返回，避免对已算好的差分特征做二次全局平滑导致相位滞后。
        return features
    
    def _add_gaussian_noise(self, features: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(features) * self.noise_std
        return features + noise
    
    def _random_occlusion(self, features: torch.Tensor) -> torch.Tensor:
        """随机遮挡"""
        mask = torch.bernoulli(
            torch.full(features.shape[:1], 1 - self.occlusion_prob)
        ).unsqueeze(-1)
        
        return features * mask


class NormalizeTransform:
    """归一化变换"""
    
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        self.mean = mean if mean is not None else torch.zeros(8)
        self.std = std if std is not None else torch.ones(8)
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / (self.std + 1e-8)
    
    def fit(self, data: torch.Tensor):
        """从数据计算均值和标准差"""
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)
