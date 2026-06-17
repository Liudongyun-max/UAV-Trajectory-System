#!/usr/bin/env encoding
from __future__ import annotations

import pytest
import torch
from training.models.gru_distance_stabilizer import GRUDistanceStabilizer


def test_gru_causality():
    # 验证单向 GRU 的因果性 (Causality): 
    # 如果我们将序列后半段的输入做修改，前向传播中，前半段输出应该保持完全不变！
    # 这证明了模型在前向推理时没有“回看”未来帧，即严格符合单向实时部署因果性。
    model = GRUDistanceStabilizer(input_dim=8, hidden_dim=16)
    model.eval()
    
    # 输入 1: 基础序列 [Batch=1, Seq_len=25, dim=8]
    x1 = torch.randn(1, 25, 8)
    
    # 输入 2: 将后 10 帧替换为不同的噪点
    x2 = x1.clone()
    x2[0, 15:, :] = torch.randn(1, 10, 8)
    
    with torch.no_grad():
        _, seq_pred1 = model(x1)
        _, seq_pred2 = model(x2)
        
    # 断言第 0 到 14 帧的预测值应该完全相同 (容许极小浮点误差)
    diff = torch.abs(seq_pred1[0, :15, 0] - seq_pred2[0, :15, 0]).max().item()
    assert diff < 1e-6, f"GRU causality violated! Future inputs changed history prediction. Max diff: {diff}"
