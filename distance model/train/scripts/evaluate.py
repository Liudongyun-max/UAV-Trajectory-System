"""
评估入口脚本

用法:
    python scripts/evaluate.py --model models/gru_baseline_final.pth --data_dir export/input
"""
import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from training.models.gru_trajectory import UAVTrajectoryNet
from training.data.splitter import EpisodeSplitter
from training.data.dataloader import create_dataloaders
from training.evaluators.classification_eval import ClassificationEvaluator
from training.evaluators.regression_eval import RegressionEvaluator
from training.evaluators.visualizer import MetricsVisualizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate UAV Trajectory Model")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (val/test)")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()
    
    from training.utils.device import get_device
    from training.utils.seed import set_seed
    
    set_seed(42)
    device = get_device(args.device)
    
    model = UAVTrajectoryNet.load_from_checkpoint(args.model, device=str(device))
    print(f"Model loaded from {args.model}")
    
    # Load config from checkpoint for self-adaptation
    checkpoint = torch.load(args.model, map_location="cpu")
    model_config = checkpoint.get("model_config", {})
    input_dim = model_config.get("input_dim", model.input_dim)
    future_steps = model_config.get("future_steps", model.future_steps)
    seq_len = model_config.get("seq_len", 20)
    
    splitter = EpisodeSplitter(root_dir=args.data_dir)
    train_ids, val_ids, test_ids = splitter.split()
    
    if args.split == "val":
        eval_ids = val_ids
    else:
        eval_ids = test_ids
    
    _, eval_loader, _ = create_dataloaders(
        root_dir=args.data_dir,
        train_ids=[],
        val_ids=eval_ids,
        test_ids=None,
        batch_size=64,
        seq_len=seq_len,
        future_steps=future_steps,
        input_dim=input_dim,
        smooth=True,
        smooth_window=5,
        smooth_polyorder=2,
    )
    
    print(f"Evaluating on {len(eval_ids)} episodes...")
    
    all_logits = []
    all_labels = []
    all_pred_offsets = []
    all_gt_offsets = []
    
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            features = batch["history_features"].to(device)
            labels = batch["label"]
            gt_offsets = batch["future_offsets"]
            
            logits, pred_offsets = model(features)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_pred_offsets.append(pred_offsets.cpu())
            all_gt_offsets.append(gt_offsets)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_pred_offsets = torch.cat(all_pred_offsets)
    all_gt_offsets = torch.cat(all_gt_offsets)
    
    cls_evaluator = ClassificationEvaluator()
    cls_metrics = cls_evaluator.compute(all_logits, all_labels)
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {cls_metrics['accuracy']:.4f}")
    print(f"  Precision: {cls_metrics['precision']:.4f}")
    print(f"  Recall:    {cls_metrics['recall']:.4f}")
    print(f"  F1:        {cls_metrics['f1']:.4f}")
    
    reg_evaluator = RegressionEvaluator()
    reg_metrics = reg_evaluator.compute(all_pred_offsets, all_gt_offsets)
    print(f"\nRegression Metrics:")
    print(f"  ADE: {reg_metrics['ade']:.6f}")
    print(f"  FDE: {reg_metrics['fde']:.6f}")
    
    if args.visualize:
        visualizer = MetricsVisualizer(save_dir="logs/eval_figures")
        
        visualizer.plot_confusion_matrix(
            cls_metrics["tp"], cls_metrics["tn"],
            cls_metrics["fp"], cls_metrics["fn"]
        )
        
        print(f"\nVisualizations saved to logs/eval_figures/")


if __name__ == "__main__":
    main()
