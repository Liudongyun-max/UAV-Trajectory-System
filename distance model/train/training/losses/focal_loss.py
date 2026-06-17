"""Focal Loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification
    
    用于处理类别不平衡问题
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        初始化Focal Loss
        
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
            reduction: 损失归约方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            logits: [B, 1] 分类logit
            targets: [B, 1] 标签
        
        Returns:
            损失值
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        
        probs = torch.sigmoid(logits)
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
