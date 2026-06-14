"""指标可视化器"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class MetricsVisualizer:
    """训练指标可视化器"""
    
    def __init__(self, save_dir: str = "logs/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self, history: Dict[str, list], save_name: str = "training_curves.png"):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history["val_acc"], label="Val Accuracy")
        axes[1].plot(history["val_f1"], label="Val F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Validation Metrics")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
    
    def plot_confusion_matrix(
        self,
        tp: int, tn: int, fp: int, fn: int,
        save_name: str = "confusion_matrix.png"
    ):
        """绘制混淆矩阵"""
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Noise", "UAV"])
        ax.set_yticklabels(["Noise", "UAV"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
        
        plt.colorbar(im)
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
    
    def plot_trajectory_examples(
        self,
        history_trajs: List[np.ndarray],
        future_trajs: List[np.ndarray],
        labels: List[int],
        save_name: str = "trajectory_examples.png"
    ):
        """绘制轨迹示例"""
        n_examples = min(4, len(history_trajs))
        
        fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
        if n_examples == 1:
            axes = [axes]
        
        for i in range(n_examples):
            ax = axes[i]
            
            hist = history_trajs[i]
            fut = future_trajs[i]
            label = labels[i]
            
            ax.plot(hist[:, 0], hist[:, 1], "b-o", markersize=2, label="History")
            ax.plot(fut[:, 0], fut[:, 1], "r-o", markersize=2, label="Future")
            
            color = "green" if label == 1 else "red"
            ax.set_title(f"{'UAV' if label == 1 else 'Noise'}", color=color)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
