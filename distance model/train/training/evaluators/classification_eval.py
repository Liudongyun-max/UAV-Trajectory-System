"""分类评估器"""
import torch
import numpy as np
from typing import Dict


class ClassificationEvaluator:
    """分类指标评估器"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            logits: [N, 1] 预测logit
            labels: [N, 1] 真实标签
        
        Returns:
            指标字典
        """
        probs = torch.sigmoid(logits).numpy().flatten()
        targets = labels.numpy().flatten()
        
        preds = (probs >= self.threshold).astype(float)
        
        tp = np.sum((preds == 1) & (targets == 1))
        tn = np.sum((preds == 0) & (targets == 0))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
    
    def compute_per_class(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """计算每类指标"""
        probs = torch.sigmoid(logits).numpy().flatten()
        targets = labels.numpy().flatten()
        
        metrics = {}
        
        for class_id in [0, 1]:
            class_name = "noise" if class_id == 0 else "uav"
            
            tp = np.sum((probs >= self.threshold) & (targets == class_id))
            fp = np.sum((probs >= self.threshold) & (targets != class_id))
            fn = np.sum((probs < self.threshold) & (targets == class_id))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        
        return metrics
