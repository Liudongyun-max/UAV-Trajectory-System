#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401
from training.utils.distance_metrics import compute_metrics, apply_kalman_filter, apply_ema


def evaluate_baselines(cfg: dict) -> None:
    out_dir = Path("outputs") / cfg["experiment"]["name"]
    seq_path = out_dir / "distance_sequences.npz"
    
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {seq_path}. Run build_distance_sequences.py first.")
        
    data = np.load(seq_path, allow_pickle=True)
    
    # 提取 fold 4 (测试集)
    fold = 4
    if f"fold_{fold}_meta" not in data:
        print("Fold 4 test set not found. Using Fold 3 or first available fold for comparison.")
        # 兜底
        for key in data.files:
            if "meta" in key:
                fold = int(key.split("_")[1])
                break
                
    meta_str = data[f"fold_{fold}_meta"][0]
    meta_list = json.loads(meta_str)
    y_true = data[f"fold_{fold}_y"].reshape(-1)
    
    n_samples = len(meta_list)
    print(f"Loaded {n_samples} test sequences from Fold {fold} for temporal baseline comparison.")
    
    raw_vals = []
    med5_vals = []
    med25_vals = []
    ema_vals = []
    kf_vals = []
    
    t_raw = 0.0
    t_med5 = 0.0
    t_med25 = 0.0
    t_ema = 0.0
    t_kf = 0.0
    
    for meta in meta_list:
        raw_seq = np.array(meta["raw_preds"], dtype=np.float32)
        missing_seq = np.array(meta["is_missing"], dtype=np.float32)
        
        # 1. Raw prediction (last frame)
        t0 = time.perf_counter()
        raw_val = raw_seq[-1]
        t_raw += (time.perf_counter() - t0)
        raw_vals.append(raw_val)
        
        # 2. Median 5
        t0 = time.perf_counter()
        med5_val = np.median(raw_seq[-5:])
        t_med5 += (time.perf_counter() - t0)
        med5_vals.append(med5_val)
        
        # 3. Median 25
        t0 = time.perf_counter()
        med25_val = np.median(raw_seq)
        t_med25 += (time.perf_counter() - t0)
        med25_vals.append(med25_val)
        
        # 4. EMA
        t0 = time.perf_counter()
        ema_seq = apply_ema(raw_seq, alpha=0.15)
        ema_val = ema_seq[-1]
        t_ema += (time.perf_counter() - t0)
        ema_vals.append(ema_val)
        
        # 5. OneDim Kalman Filter
        t0 = time.perf_counter()
        kf_seq = apply_kalman_filter(raw_seq, missing_seq, q=0.01, r=0.5)
        kf_val = kf_seq[-1]
        t_kf += (time.perf_counter() - t0)
        kf_vals.append(kf_val)

    y_true = np.array(y_true, dtype=np.float32)
    
    results = {}
    models = {
        "raw": (raw_vals, t_raw),
        "median5": (med5_vals, t_med5),
        "median25": (med25_vals, t_med25),
        "ema": (ema_vals, t_ema),
        "kalman": (kf_vals, t_kf)
    }
    
    for name, (preds, elapsed) in models.items():
        preds_arr = np.array(preds, dtype=np.float32)
        met = compute_metrics(y_true, preds_arr)
        met["cpu_time_us"] = (elapsed / n_samples) * 1e6 # 每样本微秒数
        results[name] = met
        
    # 保存结果
    with (out_dir / "stability_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    # 写入 markdown 格式比对
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "model_comparison.md").open("w", encoding="utf-8") as f:
        f.write("# 时序稳定模型与平滑算法对比报告\n\n")
        f.write("| 算法 | MAE (m) | RMSE (m) | P95绝对误差 (m) | 预测标准差 (m) | 最大跳变 (m) | P95跳变 (m) | CPU耗时/每样本 (us) |\n")
        f.write("|------|---------|----------|-----------------|----------------|--------------|-------------|---------------------|\n")
        for name, met in results.items():
            f.write(
                f"| {name:10s} | {met['mae_m']:.4f} | {met['rmse_m']:.4f} | {met['p95_abs_m']:.4f} | "
                f"{met['pred_std_m']:.4f} | {met['max_jump_m']:.4f} | {met['p95_jump_m']:.4f} | {met['cpu_time_us']:.2f} |\n"
            )
            
    print("\n--- Temporal Baselines Evaluation ---")
    for name, met in results.items():
        print(f"Model {name:10s} - MAE: {met['mae_m']:.4f}m, RMSE: {met['rmse_m']:.4f}m, Max Jump: {met['max_jump_m']:.4f}m")
        
    print(f"\nEvaluation reports saved to {out_dir / 'reports'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--models", default="raw,median5,median25,ema,kalman")
    args = p.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    evaluate_baselines(cfg)


if __name__ == "__main__":
    main()
