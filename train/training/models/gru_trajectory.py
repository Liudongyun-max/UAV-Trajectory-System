"""GRU轨迹识别模型"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class UAVTrajectoryNet(nn.Module):
    """
    无人机轨迹识别网络
    
    架构:
        Input [B, 20, 8]
            |
        GRU (input=8, hidden=32, layers=1)
            |
        Hidden [B, 32]
            |
        +-- Classifier: Linear(32->16) -> ReLU -> Linear(16->1) -> Logit
        +-- Predictor: Linear(32->32) -> ReLU -> Linear(32->10) -> [B, 5, 2]
    
    输出:
        logits: [B, 1]      分类logit (sigmoid后为UAV概率)
        future_offsets: [B, 5, 2]  未来5帧偏移
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        num_layers: int = 1,
        future_steps: int = 5,
        dropout: float = 0.1,
    ):
        """
        初始化模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: GRU隐藏层维度
            num_layers: GRU层数
            future_steps: 未来预测步数
            dropout: Dropout概率
        """
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
        """Xavier初始化"""
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
        """
        前向传播
        
        Args:
            x: [B, T, F] 输入特征 (B=batch, T=20, F=8)
        
        Returns:
            logits: [B, 1] 分类logit
            future_offsets: [B, 5, 2] 未来偏移
        """
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
        """
        预测UAV概率
        
        Args:
            x: [B, T, F] 输入特征
        
        Returns:
            proba: [B, 1] UAV概率
        """
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict_with_confidence(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        带置信度的预测
        
        Args:
            x: [B, T, F] 输入特征
            threshold: 分类阈值
        
        Returns:
            包含预测结果的字典
        """
        logits, future_offsets = self.forward(x)
        proba = torch.sigmoid(logits)
        
        predictions = (proba >= threshold).float()
        
        return {
            "logits": logits,
            "proba": proba,
            "predictions": predictions,
            "future_offsets": future_offsets,
        }
    
    def get_config(self) -> Dict:
        """获取模型配置"""
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
        """
        从检查点加载模型
        
        Args:
            path: 检查点路径
            device: 设备
        
        Returns:
            加载后的模型
        """
        checkpoint = torch.load(path, map_location=device)
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
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
