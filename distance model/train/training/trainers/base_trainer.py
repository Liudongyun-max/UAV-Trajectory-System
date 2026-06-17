"""训练器基类"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path


class BaseTrainer(ABC):
    """
    训练器基类
    
    定义训练器的基本接口
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = "cuda",
        log_dir: str = "logs",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_epoch = 0
        self.best_metric = float("inf")
    
    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
    ) -> Dict[str, list]:
        """训练主循环"""
        pass
    
    @abstractmethod
    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        pass
    
    @abstractmethod
    def _validate_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        pass
    
    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        
        path = self.log_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.log_dir / "checkpoints" / "best_model.pth"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
