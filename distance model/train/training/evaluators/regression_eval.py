"""回归评估器"""
import torch
import numpy as np
from typing import Dict


class RegressionEvaluator:
    """回归指标评估器"""
    
    def __init__(self, image_width: float = 2560.0, image_height: float = 1440.0):
        self.image_width = image_width
        self.image_height = image_height
        
    def compute(
        self,
        pred_offsets: torch.Tensor,
        gt_offsets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        计算回归指标
        
        Args:
            pred_offsets: [N, T, 2] 预测偏移
            gt_offsets: [N, T, 2] 真实偏移
        
        Returns:
            指标字典
        """
        pred = pred_offsets.numpy().copy()
        gt = gt_offsets.numpy().copy()
        
        # Denormalize relative coordinates back to actual pixel space
        pred[..., 0] *= self.image_width
        pred[..., 1] *= self.image_height
        
        gt[..., 0] *= self.image_width
        gt[..., 1] *= self.image_height
        
        errors = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))
        
        ade = np.mean(errors)
        fde = np.mean(errors[:, -1])
        
        per_step_ade = np.mean(errors, axis=0)
        
        return {
            "ade": float(ade),
            "fde": float(fde),
            "per_step_ade": per_step_ade.tolist(),
        }
