"""损失函数测试"""
import pytest
import torch
from training.losses.combined_loss import CombinedTrajectoryLoss
from training.losses.focal_loss import FocalLoss


class TestCombinedTrajectoryLoss:
    """组合损失测试"""
    
    def test_forward(self):
        loss_fn = CombinedTrajectoryLoss(cls_weight=1.0, reg_weight=0.5)
        
        logits = torch.randn(4, 1, requires_grad=True)
        labels = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)
        pred_offsets = torch.randn(4, 5, 2, requires_grad=True)
        gt_offsets = torch.randn(4, 5, 2)
        
        losses = loss_fn(logits, labels, pred_offsets, gt_offsets)
        
        assert "total_loss" in losses
        assert "cls_loss" in losses
        assert "reg_loss" in losses
        
        assert losses["total_loss"].requires_grad
        assert losses["total_loss"].shape == ()
    
    def test_no_positive_samples(self):
        loss_fn = CombinedTrajectoryLoss(cls_weight=1.0, reg_weight=0.5)
        
        logits = torch.randn(4, 1)
        labels = torch.zeros(4, 1)
        pred_offsets = torch.randn(4, 5, 2)
        gt_offsets = torch.randn(4, 5, 2)
        
        losses = loss_fn(logits, labels, pred_offsets, gt_offsets)
        
        assert losses["reg_loss"] == 0.0


class TestFocalLoss:
    """Focal Loss测试"""
    
    def test_forward(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        logits = torch.randn(4, 1, requires_grad=True)
        targets = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)
        
        loss = loss_fn(logits, targets)
        
        assert loss.requires_grad
        assert loss.shape == ()
    
    def test_no_reduction(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        
        logits = torch.randn(4, 1)
        targets = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)
        
        loss = loss_fn(logits, targets)
        
        assert loss.shape == (4, 1)
