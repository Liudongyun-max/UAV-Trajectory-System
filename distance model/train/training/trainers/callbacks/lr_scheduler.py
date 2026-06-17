"""学习率调度回调"""
from typing import Dict


class LRSchedulerCallback:
    """学习率调度回调"""
    
    def __init__(self, patience: int = 10, factor: float = 0.5):
        self.patience = patience
        self.factor = factor
        self.counter = 0
        self.best_metric = float("inf")
    
    def on_train_begin(self, trainer):
        import torch.optim.lr_scheduler as lr_scheduler
        
        trainer.scheduler = lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
        )
    
    def on_epoch_begin(self, epoch: int, trainer):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict, trainer):
        val_loss = metrics["val"]["total_loss"]
        trainer.scheduler.step(val_loss)
    
    def on_train_end(self, trainer):
        pass
