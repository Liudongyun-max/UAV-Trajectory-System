#!/usr/bin/env python3
from __future__ import annotations

import copy
import torch
from torch.utils.data import DataLoader
from ..losses.distance_stability_loss import DistanceStabilityLoss


class DistanceGRUTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.00001,
        loss_type: str = "huber",
        stability_weight: float = 0.05,
        early_stopping_patience: int = 12
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.loss_fn = DistanceStabilityLoss(loss_type=loss_type, stability_weight=stability_weight)
        self.patience = early_stopping_patience
        
        print(f"DistanceGRUTrainer initialized on device: {self.device}")

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_reg = 0.0
        total_stab = 0.0
        
        for x, y, _, _, _ in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            pred_last, pred_seq = self.model(x)
            
            loss, reg, stab = self.loss_fn(pred_last, pred_seq, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(x)
            total_reg += reg.item() * len(x)
            total_stab += stab.item() * len(x)
            
        n = len(dataloader.dataset)
        return total_loss / n, total_reg / n, total_stab / n

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_reg = 0.0
        total_stab = 0.0
        
        with torch.no_grad():
            for x, y, _, _, _ in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred_last, pred_seq = self.model(x)
                loss, reg, stab = self.loss_fn(pred_last, pred_seq, y)
                
                total_loss += loss.item() * len(x)
                total_reg += reg.item() * len(x)
                total_stab += stab.item() * len(x)
                
        n = len(dataloader.dataset)
        return total_loss / n, total_reg / n, total_stab / n

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100) -> list[dict]:
        best_loss = float("inf")
        best_weights = None
        patience_counter = 0
        history = []
        
        for epoch in range(1, epochs + 1):
            train_loss, train_reg, train_stab = self.train_epoch(train_loader)
            val_loss, val_reg, val_stab = self.evaluate(val_loader)
            
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_reg": train_reg,
                "train_stability": train_stab,
                "val_loss": val_loss,
                "val_reg": val_reg,
                "val_stability": val_stab,
            })
            
            # 早停与最佳权重保存
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} - Train Loss: {train_loss:.4f} (Reg: {train_reg:.4f}, Stab: {train_stab:.4f}) | Val Loss: {val_loss:.4f} (Reg: {val_reg:.4f}, Stab: {val_stab:.4f})")
                
            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_loss:.4f}")
                break
                
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        return history
