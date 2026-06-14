"""组合损失函数"""
import torch
import torch.nn as nn
from typing import Dict


class CombinedTrajectoryLoss(nn.Module):
    """
    组合损失: 分类 + 回归
    
    total_loss = cls_weight * BCEWithLogitsLoss 
               + reg_weight * SmoothL1Loss (仅正样本)
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 0.5,
        pos_weight: float = 1.0,
    ):
        """
        初始化损失函数
        
        Args:
            cls_weight: 分类损失权重
            reg_weight: 回归损失权重
            pos_weight: 正样本权重 (处理类别不平衡)
        """
        super().__init__()
        
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        self.cls_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.reg_loss = nn.SmoothL1Loss()
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pred_offsets: torch.Tensor,
        gt_offsets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            logits: [B, 1] 分类logit
            labels: [B, 1] 标签
            pred_offsets: [B, 5, 2] 预测偏移
            gt_offsets: [B, 5, 2] 真实偏移
        
        Returns:
            包含各项损失的字典
        """
        cls_loss = self.cls_loss(logits, labels)
        
        pos_mask = labels.squeeze(-1) > 0.5
        
        if pos_mask.any():
            reg_loss = self.reg_loss(
                pred_offsets[pos_mask],
                gt_offsets[pos_mask]
            )
        else:
            reg_loss = torch.tensor(0.0, device=logits.device)
        
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
        }
