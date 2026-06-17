#!/usr/bin/env python3
from __future__ import annotations

import pytest
import torch
from training.models.gru_distance_stabilizer import GRUDistanceStabilizer


def test_distance_gru_unidirectional_and_dimensions():
    model = GRUDistanceStabilizer(input_dim=16, hidden_dim=32, num_layers=1)
    
    # 断言是单向 GRU
    assert model.gru.bidirectional is False, "GRU must be unidirectional!"
    
    # 模拟输入 [Batch=4, Sequence=25, features=16]
    x = torch.zeros(4, 25, 16)
    last_pred, seq_pred = model(x)
    
    # 断言输出维度正确
    assert last_pred.shape == (4, 1), "Last pred shape must be [Batch, 1]"
    assert seq_pred.shape == (4, 25, 1), "Sequence pred shape must be [Batch, Sequence, 1]"
    
    # 验证输入维度动态感知
    model_12 = GRUDistanceStabilizer(input_dim=12)
    assert model_12.gru.input_size == 12
