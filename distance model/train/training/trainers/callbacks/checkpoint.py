"""模型检查点回调"""
import torch
from pathlib import Path
from typing import Dict


class CheckpointCallback:
    """模型保存回调"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.best_metric = float("inf")
    
    def on_train_begin(self, trainer):
        pass
    
    def on_epoch_begin(self, epoch: int, trainer):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict, trainer):
        val_loss = metrics["val"]["total_loss"]
        
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            self._save_checkpoint(epoch, metrics, trainer, is_best=True)
        
        self._save_checkpoint(epoch, metrics, trainer, is_best=False)
    
    def on_train_end(self, trainer):
        pass
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, trainer, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics": metrics,
            "config": trainer.config,
        }
        
        if is_best:
            path = self.log_dir / "checkpoints" / "best_model.pth"
        else:
            path = self.log_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
