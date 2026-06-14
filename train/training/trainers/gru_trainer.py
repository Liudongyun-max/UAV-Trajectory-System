"""GRU训练器"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import logging
from pathlib import Path

from .base_trainer import BaseTrainer
from ..models.gru_trajectory import UAVTrajectoryNet
from ..losses.combined_loss import CombinedTrajectoryLoss
from ..evaluators.classification_eval import ClassificationEvaluator
from .callbacks import CheckpointCallback, EarlyStoppingCallback, LRSchedulerCallback, MetricsLogger


class GRUTrainer(BaseTrainer):
    """
    GRU训练器
    
    功能:
    - 训练循环
    - 验证循环
    - 模型保存
    - 早停
    - 学习率调度
    """
    
    def __init__(
        self,
        model: UAVTrajectoryNet,
        config: dict,
        device: str = "cuda",
        log_dir: str = "logs",
    ):
        super().__init__(model, config, device, log_dir)
        
        self.criterion = CombinedTrajectoryLoss(
            cls_weight=config.get("cls_weight", 1.0),
            reg_weight=config.get("reg_weight", 0.5),
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )
        
        self.callbacks = [
            CheckpointCallback(log_dir),
            EarlyStoppingCallback(patience=config.get("patience", 15)),
            LRSchedulerCallback(),
            MetricsLogger(log_dir),
        ]
        
        self.evaluator = ClassificationEvaluator()
        
        self.logger = logging.getLogger(__name__)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
    ) -> Dict[str, list]:
        """
        训练主循环
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
        
        Returns:
            训练历史
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }
        
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self)
            
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            history["train_loss"].append(train_metrics["total_loss"])
            history["val_loss"].append(val_metrics["total_loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])
            
            metrics = {
                "train": train_metrics,
                "val": val_metrics,
            }
            
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics, self)
            
            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f} - "
                f"Val Loss: {val_metrics['total_loss']:.4f} - "
                f"Val Acc: {val_metrics['accuracy']:.4f} - "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            if val_metrics["total_loss"] < self.best_metric:
                self.best_metric = val_metrics["total_loss"]
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Check for early stopping
            if any(getattr(callback, "should_stop", False) for callback in self.callbacks):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        return history
    
    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            features = batch["history_features"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(-1)
            gt_offsets = batch["future_offsets"].to(self.device)
            
            logits, pred_offsets = self.model(features)
            
            loss_dict = self.criterion(logits, labels, pred_offsets, gt_offsets)
            
            self.optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("grad_clip", 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss_dict["total_loss"].item()
            n_batches += 1
        
        return {"total_loss": total_loss / n_batches}
    
    def _validate_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        all_logits = []
        all_labels = []
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                features = batch["history_features"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(-1)
                gt_offsets = batch["future_offsets"].to(self.device)
                
                logits, pred_offsets = self.model(features)
                
                loss_dict = self.criterion(logits, labels, pred_offsets, gt_offsets)
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                
                total_loss += loss_dict["total_loss"].item()
                n_batches += 1
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        metrics = self.evaluator.compute(all_logits, all_labels)
        metrics["total_loss"] = total_loss / n_batches
        
        return metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """在测试集上评估"""
        return self._validate_epoch(test_loader, -1)
