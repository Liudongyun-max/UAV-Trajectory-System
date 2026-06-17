#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn


class GRUDistanceStabilizer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        # 正式系统只允许单向 GRU
        if bidirectional:
            raise ValueError("Bidirectional GRU is not allowed for live deployment stabilizer models.")
            
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [Batch, Sequence_length, input_dim]
        # out: [Batch, Sequence_length, hidden_dim]
        out, _ = self.gru(x)
        
        # 获得整个时序序列每一帧的预测, 用于计算帧间跳变平滑正则
        seq_pred = self.fc(out) # [Batch, Sequence_length, 1]
        
        # 提取最后一帧用于作为当前窗口的稳定测距输出 (target_mode: last_frame)
        last_pred = seq_pred[:, -1, :] # [Batch, 1]
        
        return last_pred, seq_pred
