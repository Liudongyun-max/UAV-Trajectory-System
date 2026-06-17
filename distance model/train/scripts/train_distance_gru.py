#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from training.data.distance_sequence_dataset import DistanceSequenceDataset
from training.models.gru_distance_stabilizer import GRUDistanceStabilizer
from training.trainers.distance_gru_trainer import DistanceGRUTrainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    out_dir = Path("outputs") / cfg["experiment"]["name"]
    seq_path = out_dir / "distance_sequences.npz"
    schema_path = out_dir / "feature_schema.json"
    
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {seq_path}. Run build_distance_sequences.py first.")
        
    # 读取 schema 感知 input_dim
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    input_dim = schema["input_dim"]
    
    # 载入数据
    data = np.load(seq_path, allow_pickle=True)
    
    # 将 fold 载入为 Dataset
    folds_dataset = {}
    for f_id in range(5):
        if f"fold_{f_id}_x" in data:
            fold_dict = {
                "x": data[f"fold_{f_id}_x"],
                "y": data[f"fold_{f_id}_y"],
                "missing_mask": data[f"fold_{f_id}_missing"],
                "interpolated_mask": data[f"fold_{f_id}_interpolated"],
                # 元数据还原
                "meta": json.loads(data[f"fold_{f_id}_meta"][0])
            }
            folds_dataset[f_id] = DistanceSequenceDataset(fold_dict)
            
    print(f"Loaded folds dataset for training. Input dimension: {input_dim}")
    
    # 5折 GroupKFold 循环训练
    all_history = []
    best_val_losses = {}
    
    # 创建 checkpoints 目录
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    patience = cfg["training"]["early_stopping_patience"]
    
    loss_type = cfg["loss"]["type"]
    stab_weight = cfg["loss"]["stability_weight"]
    
    # 动态参数
    hidden_dim = cfg["model"]["hidden_dim"]
    num_layers = cfg["model"]["num_layers"]
    bidirectional = cfg["model"].get("bidirectional", False)

    for val_fold in range(5):
        print(f"\n================ Training Fold {val_fold} (Validation) ================")
        
        # 收集训练集
        train_x = []
        train_y = []
        train_miss = []
        train_interp = []
        train_meta = []
        
        for f_id in range(5):
            if f_id == val_fold:
                continue
            if folds_dataset[f_id] is not None:
                train_x.append(folds_dataset[f_id].x.numpy())
                train_y.append(folds_dataset[f_id].y.numpy())
                train_miss.append(folds_dataset[f_id].missing_mask.numpy())
                train_interp.append(folds_dataset[f_id].interpolated_mask.numpy())
                train_meta.extend(folds_dataset[f_id].meta)
                
        train_dict = {
            "x": np.concatenate(train_x, axis=0),
            "y": np.concatenate(train_y, axis=0),
            "missing_mask": np.concatenate(train_miss, axis=0),
            "interpolated_mask": np.concatenate(train_interp, axis=0),
            "meta": train_meta
        }
        
        train_ds = DistanceSequenceDataset(train_dict)
        val_ds = folds_dataset[val_fold]
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # 定义模型与训练器
        model = GRUDistanceStabilizer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        
        trainer = DistanceGRUTrainer(
            model=model,
            learning_rate=lr,
            weight_decay=wd,
            loss_type=loss_type,
            stability_weight=stab_weight,
            early_stopping_patience=patience
        )
        
        hist = trainer.fit(train_loader, val_loader, epochs=epochs)
        
        # 提取最小验证 loss 作为评估
        val_losses = [h["val_loss"] for h in hist]
        best_val_losses[val_fold] = min(val_losses)
        
        # 保存这折的模型
        torch.save(model.state_dict(), ckpt_dir / f"best_fold_{val_fold}.pth")
        
        # 记录历史
        for h in hist:
            h["val_fold"] = val_fold
            all_history.append(h)
            
        # 最后一折 (Fold 4 作为验证折) 拟合的模型直接拷贝为 best.pth / last.pth 做生产部署
        if val_fold == 4:
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            torch.save(model.state_dict(), ckpt_dir / "last.pth")
            print("Saved Fold 4 best checkpoint as production production weights (best.pth).")

    # 保存训练历史 CSV
    pd.DataFrame(all_history).to_csv(out_dir / "training_history.csv", index=False)
    
    print("\n--- 5-Fold Cross Validation Results ---")
    for f_id, v_loss in best_val_losses.items():
        print(f"Fold {f_id} Best Val Loss (Huber+Stability): {v_loss:.4f}")
        
    print(f"Training completed. Outputs saved to {out_dir}")


import json
if __name__ == "__main__":
    main()
