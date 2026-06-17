#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn


class DistanceStabilityLoss(nn.Module):
    def __init__(self, loss_type: str = "huber", stability_weight: float = 0.05):
        super().__init__()
        self.stability_weight = stability_weight
        if loss_type.lower() == "huber":
            self.reg_loss_fn = nn.HuberLoss()
        else:
            self.reg_loss_fn = nn.MSELoss()

    def forward(
        self,
        pred_last: torch.Tensor,
        pred_seq: torch.Tensor,
        target_last: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. 基础测距 Huber 损失
        reg_loss = self.reg_loss_fn(pred_last, target_last)
        
        # 2. 帧间波动惩罚 (稳定性正则)
        # pred_seq 形状为 [Batch, Seq_len, 1]
        diffs = pred_seq[:, 1:, :] - pred_seq[:, :-1, :]
        stability_loss = torch.mean(torch.abs(diffs))
        
        # 3. 组合损失
        total_loss = reg_loss + self.stability_weight * stability_loss
        
        return total_loss, reg_loss, stability_loss
