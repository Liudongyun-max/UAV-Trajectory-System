"""指标日志回调"""
import json
from pathlib import Path
from typing import Dict


class MetricsLogger:
    """指标日志回调"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.metrics_history = []
    
    def on_train_begin(self, trainer):
        self.metrics_history = []
    
    def on_epoch_begin(self, epoch: int, trainer):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict, trainer):
        record = {
            "epoch": epoch,
            "train_loss": metrics["train"]["total_loss"],
            "val_loss": metrics["val"]["total_loss"],
            "val_acc": metrics["val"]["accuracy"],
            "val_f1": metrics["val"]["f1"],
        }
        
        self.metrics_history.append(record)
        
        log_path = self.log_dir / "metrics.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def on_train_end(self, trainer):
        pass
