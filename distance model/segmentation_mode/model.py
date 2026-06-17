#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRU轨迹预测网络定义 (独立推理解耦版)
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict

class UAVTrajectoryNet(nn.Module):
    """
    无人机轨迹识别与预测网络
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        num_layers: int = 1,
        future_steps: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, future_steps * 2),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"输入必须为3D张量 [B,T,F]，收到 {x.shape}")
        
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"输入特征维度必须为 {self.input_dim}，收到 {x.shape[-1]}")
        
        gru_out, hidden = self.gru(x)
        last_hidden = hidden[-1]
        
        logits = self.classifier(last_hidden)
        pred = self.predictor(last_hidden)
        future_offsets = pred.view(-1, self.future_steps, 2)
        
        return logits, future_offsets
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_config(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.gru.num_layers,
            "future_steps": self.future_steps,
            "dropout": self.classifier[2].p if len(self.classifier) > 2 else 0,
        }
    
    @staticmethod
    def load_from_checkpoint(
        path: str, 
        device: str = "cpu"
    ) -> "UAVTrajectoryNet":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = checkpoint.get("model_config", {})
        
        model = UAVTrajectoryNet(
            input_dim=config.get("input_dim", 8),
            hidden_dim=config.get("hidden_dim", 32),
            num_layers=config.get("num_layers", 1),
            future_steps=config.get("future_steps", 5),
            dropout=config.get("dropout", 0.1),
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        return model
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
