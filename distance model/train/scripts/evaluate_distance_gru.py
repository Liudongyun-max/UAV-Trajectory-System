#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from training.utils.distance_metrics import compute_metrics


def run_inference(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for x, y, _, _, _ in dataloader:
            x = x.to(device)
            pred_last, _ = model(x)
            all_preds.append(pred_last.cpu().numpy())
            all_trues.append(y.numpy())
    return np.concatenate(all_preds, axis=0).reshape(-1), np.concatenate(all_trues, axis=0).reshape(-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    out_dir = Path("outputs") / cfg["experiment"]["name"]
    seq_path = out_dir / "distance_sequences.npz"
    schema_path = out_dir / "feature_schema.json"
    
    # 动态感知输入维度
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    input_dim = schema["input_dim"]
    
    data = np.load(seq_path, allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    folds_dataset = {}
    for f_id in range(5):
        if f"fold_{f_id}_x" in data:
            fold_dict = {
                "x": data[f"fold_{f_id}_x"],
                "y": data[f"fold_{f_id}_y"],
                "missing_mask": data[f"fold_{f_id}_missing"],
                "interpolated_mask": data[f"fold_{f_id}_interpolated"],
                "meta": json.loads(data[f"fold_{f_id}_meta"][0])
            }
            folds_dataset[f_id] = DistanceSequenceDataset(fold_dict)

    # 1. 5折交叉验证 Out-of-Fold 评估
    oof_preds = []
    oof_trues = []
    oof_meta = []
    
    for val_fold in range(5):
        if folds_dataset[val_fold] is None:
            continue
            
        # 加载对应 fold 的模型
        model = GRUDistanceStabilizer(
            input_dim=input_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            bidirectional=cfg["model"].get("bidirectional", False)
        ).to(device)
        
        ckpt_path = out_dir / "checkpoints" / f"best_fold_{val_fold}.pth"
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found for fold {val_fold}. Skipping OOF inference.")
            continue
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        loader = DataLoader(folds_dataset[val_fold], batch_size=256, shuffle=False)
        preds, trues = run_inference(model, loader, device)
        
        oof_preds.append(preds)
        oof_trues.append(trues)
        oof_meta.extend(folds_dataset[val_fold].meta)

    oof_preds = np.concatenate(oof_preds, axis=0)
    oof_trues = np.concatenate(oof_trues, axis=0)
    
    # 整体指标
    overall = compute_metrics(oof_trues, oof_preds)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    with (reports_dir / "overall_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # 构造元数据评估 DataFrame
    df_eval = pd.DataFrame(oof_meta)
    df_eval["true_range_m"] = oof_trues
    df_eval["pred_range_m"] = oof_preds

    # 2. 按场景评估
    scene_metrics = []
    for scene, sub_df in df_eval.groupby("scene_profile"):
        m = compute_metrics(sub_df["true_range_m"].to_numpy(), sub_df["pred_range_m"].to_numpy())
        m["scene"] = scene
        scene_metrics.append(m)
    pd.DataFrame(scene_metrics).to_csv(reports_dir / "metrics_by_scene.csv", index=False)

    # 3. 按距离段评估 (50m 步长)
    df_eval["distance_band"] = (df_eval["nominal_distance"] // 50 * 50).astype(int)
    band_metrics = []
    for band, sub_df in df_eval.groupby("distance_band"):
        m = compute_metrics(sub_df["true_range_m"].to_numpy(), sub_df["pred_range_m"].to_numpy())
        m["distance_band"] = f"{band:03d}-{band + 49:03d}"
        band_metrics.append(m)
    pd.DataFrame(band_metrics).to_csv(reports_dir / "metrics_by_distance_band.csv", index=False)

    # 4. 按 FPS 评估
    fps_metrics = []
    for fps, sub_df in df_eval.groupby("effective_fps"):
        m = compute_metrics(sub_df["true_range_m"].to_numpy(), sub_df["pred_range_m"].to_numpy())
        m["fps"] = int(fps)
        fps_metrics.append(m)
    pd.DataFrame(fps_metrics).to_csv(reports_dir / "metrics_by_fps.csv", index=False)

    # 5. 远距离外推评估 (20-350m 训练, 351-500m 测试)
    print("\n--- Running Distance Extrapolation Experiment ---")
    all_x = []
    all_y = []
    all_miss = []
    all_interp = []
    all_meta_ext = []
    
    for f_id in range(5):
        if folds_dataset[f_id] is not None:
            all_x.append(folds_dataset[f_id].x.numpy())
            all_y.append(folds_dataset[f_id].y.numpy())
            all_miss.append(folds_dataset[f_id].missing_mask.numpy())
            all_interp.append(folds_dataset[f_id].interpolated_mask.numpy())
            all_meta_ext.extend(folds_dataset[f_id].meta)
            
    x_all = np.concatenate(all_x, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    miss_all = np.concatenate(all_miss, axis=0)
    interp_all = np.concatenate(all_interp, axis=0)
    
    # 按照最后一帧的 true_range_m (或 nominal_distance) 划分
    train_ext_idx = np.where(y_all.reshape(-1) <= 350.0)[0]
    test_ext_idx = np.where((y_all.reshape(-1) > 350.0) & (y_all.reshape(-1) <= 500.0))[0]
    
    if len(train_ext_idx) > 0 and len(test_ext_idx) > 0:
        ext_train_ds = DistanceSequenceDataset({
            "x": x_all[train_ext_idx],
            "y": y_all[train_ext_idx],
            "missing_mask": miss_all[train_ext_idx],
            "interpolated_mask": interp_all[train_ext_idx],
            "meta": [all_meta_ext[i] for i in train_ext_idx]
        })
        
        ext_test_ds = DistanceSequenceDataset({
            "x": x_all[test_ext_idx],
            "y": y_all[test_ext_idx],
            "missing_mask": miss_all[test_ext_idx],
            "interpolated_mask": interp_all[test_ext_idx],
            "meta": [all_meta_ext[i] for i in test_ext_idx]
        })
        
        ext_train_loader = DataLoader(ext_train_ds, batch_size=128, shuffle=True)
        ext_test_loader = DataLoader(ext_test_ds, batch_size=256, shuffle=False)
        
        # 重新快速训练一个外推模型 (30 epochs)
        ext_model = GRUDistanceStabilizer(
            input_dim=input_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            bidirectional=cfg["model"].get("bidirectional", False)
        ).to(device)
        
        ext_trainer = DistanceGRUTrainer(
            model=ext_model,
            learning_rate=0.001,
            loss_type=cfg["loss"]["type"],
            stability_weight=cfg["loss"]["stability_weight"],
            early_stopping_patience=10
        )
        
        print("Fitting GRU extrapolation model...")
        ext_trainer.fit(ext_train_loader, ext_test_loader, epochs=30)
        
        ext_preds, ext_trues = run_inference(ext_model, ext_test_loader, device)
        extrapolation_results = compute_metrics(ext_trues, ext_preds)
        
        with (reports_dir / "distance_extrapolation_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(extrapolation_results, f, ensure_ascii=False, indent=2)
            
        print(f"Extrapolation Model (350m->500m) - MAE: {extrapolation_results['mae_m']:.4f}m, RMSE: {extrapolation_results['rmse_m']:.4f}m")
    else:
        print("Warning: Insufficient samples for extrapolation experiment.")

    print(f"\nAll evaluations completed. Reports generated at {reports_dir}")


if __name__ == "__main__":
    main()
