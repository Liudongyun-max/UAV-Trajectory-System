"""
训练入口脚本

用法:
    python scripts/train.py --config configs/gru_baseline.yaml
    python scripts/train.py --config configs/gru_baseline.yaml --device cuda --epochs 200
"""
import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from training.config.experiment import ExperimentConfig
from training.data.splitter import EpisodeSplitter
from training.data.dataloader import create_dataloaders
from training.models.gru_trajectory import UAVTrajectoryNet
from training.trainers.gru_trainer import GRUTrainer
from training.utils.seed import set_seed
from training.utils.logger import setup_logger
from training.utils.device import get_device, get_device_info


def main():
    parser = argparse.ArgumentParser(description="Train UAV Trajectory Model")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()
    
    config = ExperimentConfig.from_yaml(args.config)
    
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.seed:
        config.seed = args.seed
    
    set_seed(config.seed)
    device = get_device(args.device)
    
    print(f"Device info: {get_device_info()}")
    print(f"Using device: {device}")
    
    logger = setup_logger(f"logs/{config.name}.log")
    
    splitter = EpisodeSplitter(
        root_dir=config.data.root_dir,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        split_by=config.data.split_by,
        seed=config.seed,
    )
    
    train_ids, val_ids, test_ids = splitter.split()
    
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=config.data.root_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=config.training.batch_size,
        seq_len=config.data.seq_len,
        future_steps=config.model.future_steps,
        input_dim=config.model.input_dim,
        negative_ratio=config.data.negative_ratio,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        smooth=config.data.smooth,
        smooth_window=config.data.smooth_window,
        smooth_polyorder=config.data.smooth_polyorder,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    model = UAVTrajectoryNet(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        future_steps=config.model.future_steps,
        dropout=config.model.dropout,
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    trainer = GRUTrainer(
        model=model,
        config={
            "learning_rate": config.training.learning_rate,
            "cls_weight": config.training.cls_weight,
            "reg_weight": config.training.reg_weight,
            "grad_clip": config.training.grad_clip,
            "patience": config.training.patience,
        },
        device=str(device),
        log_dir=f"logs/{config.name}",
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
    )
    
    if test_loader:
        test_metrics = trainer.evaluate(test_loader)
        print(f"Test metrics: {test_metrics}")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config.model.__dict__,
        "training_config": config.training.__dict__,
    }, f"models/{config.name}_final.pth")
    
    print(f"Training complete. Best val loss: {trainer.best_metric:.4f}")


if __name__ == "__main__":
    main()
