"""早停回调"""
from typing import Dict


class EarlyStoppingCallback:
    """早停回调"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float("inf")
        self.should_stop = False
    
    def on_train_begin(self, trainer):
        self.counter = 0
        self.best_metric = float("inf")
        self.should_stop = False
    
    def on_epoch_begin(self, epoch: int, trainer):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict, trainer):
        val_loss = metrics["val"]["total_loss"]
        
        if val_loss < self.best_metric - self.min_delta:
            self.best_metric = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.should_stop = True
    
    def on_train_end(self, trainer):
        pass
