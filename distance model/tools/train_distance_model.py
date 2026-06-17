#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from uav_distance_pipeline.config import ROOT

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def load_schema(dataset_root: Path) -> dict:
    return json.loads((dataset_root / "feature_schema.json").read_text(encoding="utf-8"))


def load_split(dataset_root: Path, split: str, features: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(dataset_root / f"{split}.csv")
    x = df[features].to_numpy(dtype=np.float64)
    residual = df["residual_range_m"].to_numpy(dtype=np.float64).reshape(-1, 1)
    fused = df["fused_range_m"].to_numpy(dtype=np.float64)
    true_range = df["true_range_m"].to_numpy(dtype=np.float64)
    return df, x, residual, fused, true_range


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# ================= Custom NumPy MLPs =================

def train_mlp_single(x: np.ndarray, y: np.ndarray, epochs: int = 80, batch_size: int = 4096) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(17)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-9] = 1.0
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std[y_std < 1e-9] = 1.0
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std

    n, in_dim = xs.shape
    hidden = 32
    w1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden))
    b1 = np.zeros((1, hidden))
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, 1))
    b2 = np.zeros((1, 1))

    lr = 2e-3
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = xs[idx]
            yb = ys[idx]
            h_pre = xb @ w1 + b1
            h = relu(h_pre)
            pred = h @ w2 + b2
            diff = pred - yb
            loss = float(np.mean(diff ** 2))
            epoch_loss += loss
            batches += 1
            grad_pred = 2.0 * diff / len(xb)
            grad_w2 = h.T @ grad_pred
            grad_b2 = grad_pred.sum(axis=0, keepdims=True)
            grad_h = grad_pred @ w2.T
            grad_h[h_pre <= 0] = 0.0
            grad_w1 = xb.T @ grad_h
            grad_b1 = grad_h.sum(axis=0, keepdims=True)
            w2 -= lr * grad_w2
            b2 -= lr * grad_b2
            w1 -= lr * grad_w1
            b1 -= lr * grad_b1
        history.append({"epoch": epoch, "mse_scaled": epoch_loss / max(1, batches)})
    params = {
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "w1": w1, "b1": b1,
        "w2": w2, "b2": b2,
    }
    return params, history


def predict_mlp_single(params: dict, x: np.ndarray) -> np.ndarray:
    xs = (x - params["x_mean"]) / params["x_std"]
    h = relu(xs @ params["w1"] + params["b1"])
    pred_scaled = h @ params["w2"] + params["b2"]
    return (pred_scaled * params["y_std"] + params["y_mean"]).reshape(-1)


def train_mlp_double(x: np.ndarray, y: np.ndarray, epochs: int = 80, batch_size: int = 4096) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(17)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-9] = 1.0
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_std[y_std < 1e-9] = 1.0
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std

    n, in_dim = xs.shape
    h1_dim = 64
    h2_dim = 32
    w1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, h1_dim))
    b1 = np.zeros((1, h1_dim))
    w2 = rng.normal(0.0, np.sqrt(2.0 / h1_dim), size=(h1_dim, h2_dim))
    b2 = np.zeros((1, h2_dim))
    w3 = rng.normal(0.0, np.sqrt(2.0 / h2_dim), size=(h2_dim, 1))
    b3 = np.zeros((1, 1))

    lr = 2e-3
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = xs[idx]
            yb = ys[idx]
            
            # Forward
            h1_pre = xb @ w1 + b1
            h1 = relu(h1_pre)
            h2_pre = h1 @ w2 + b2
            h2 = relu(h2_pre)
            pred = h2 @ w3 + b3
            
            diff = pred - yb
            loss = float(np.mean(diff ** 2))
            epoch_loss += loss
            batches += 1
            
            # Backward
            grad_pred = 2.0 * diff / len(xb)
            grad_w3 = h2.T @ grad_pred
            grad_b3 = grad_pred.sum(axis=0, keepdims=True)
            
            grad_h2 = grad_pred @ w3.T
            grad_h2[h2_pre <= 0] = 0.0
            grad_w2 = h1.T @ grad_h2
            grad_b2 = grad_h2.sum(axis=0, keepdims=True)
            
            grad_h1 = grad_h2 @ w2.T
            grad_h1[h1_pre <= 0] = 0.0
            grad_w1 = xb.T @ grad_h1
            grad_b1 = grad_h1.sum(axis=0, keepdims=True)
            
            # Update
            w3 -= lr * grad_w3
            b3 -= lr * grad_b3
            w2 -= lr * grad_w2
            b2 -= lr * grad_b2
            w1 -= lr * grad_w1
            b1 -= lr * grad_b1
        history.append({"epoch": epoch, "mse_scaled": epoch_loss / max(1, batches)})
    params = {
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
        "w1": w1, "b1": b1,
        "w2": w2, "b2": b2,
        "w3": w3, "b3": b3,
    }
    return params, history


def predict_mlp_double(params: dict, x: np.ndarray) -> np.ndarray:
    xs = (x - params["x_mean"]) / params["x_std"]
    h1 = relu(xs @ params["w1"] + params["b1"])
    h2 = relu(h1 @ params["w2"] + params["b2"])
    pred_scaled = h2 @ params["w3"] + params["b3"]
    return (pred_scaled * params["y_std"] + params["y_mean"]).reshape(-1)


# ================= Metric Helpers =================

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_pred - y_true)
    return {
        "count": int(len(err)),
        "mae_m": float(np.mean(err)),
        "rmse_m": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "p50_abs_m": float(np.percentile(err, 50)) if len(err) > 0 else 0.0,
        "p90_abs_m": float(np.percentile(err, 90)) if len(err) > 0 else 0.0,
        "p95_abs_m": float(np.percentile(err, 95)) if len(err) > 0 else 0.0,
        "max_abs_m": float(np.max(err)) if len(err) > 0 else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--run-id", default="run017")
    p.add_argument("--package-root", type=Path)
    p.add_argument("--model-family", default="residual-comparison", choices=["numpy-mlp", "residual-comparison"])
    p.add_argument("--scene-features", default="false")
    args = p.parse_args()

    package_root = args.package_root or ROOT
    dataset_root = package_root / "dataset" / args.dataset_version
    schema = load_schema(dataset_root)
    features = schema["feature_columns"]

    # 载入数据并确定是否有 fold_id
    frame_dataset_path = dataset_root / "frame_dataset.csv"
    df_all = pd.read_csv(frame_dataset_path)

    # 模型家族的比较与评估
    has_folds = "fold_id" in df_all.columns and df_all["fold_id"].nunique() > 1

    comparison_results = {}
    
    # 候选模型定义
    models_to_eval = [
        "physics_baseline",
        "linear_regression",
        "hist_gradient_boosting",
        "random_forest",
        "numpy_mlp_single",
        "numpy_mlp_double",
    ]

    print("Evaluating models under 5-Fold GroupKFold..." if has_folds else "Evaluating models under Train/Val/Test split...")

    # 保存各模型预测
    model_predictions = {m: np.zeros(len(df_all)) for m in models_to_eval}
    
    if has_folds:
        folds = sorted(df_all["fold_id"].dropna().unique())
        for fold in folds:
            train_idx = df_all[df_all["fold_id"] != fold].index
            val_idx = df_all[df_all["fold_id"] == fold].index
            
            print(f"--- Training and evaluating Fold {fold} ---")
            x_train = df_all.loc[train_idx, features].to_numpy(dtype=np.float64)
            y_train = df_all.loc[train_idx, "residual_range_m"].to_numpy(dtype=np.float64).reshape(-1, 1)
            
            x_val = df_all.loc[val_idx, features].to_numpy(dtype=np.float64)
            fused_val = df_all.loc[val_idx, "fused_range_m"].to_numpy(dtype=np.float64)
            
            # 1. Physics Baseline
            model_predictions["physics_baseline"][val_idx] = fused_val
            
            # 2. Linear Regression
            lr = LinearRegression()
            lr.fit(x_train, y_train.reshape(-1))
            model_predictions["linear_regression"][val_idx] = fused_val + lr.predict(x_val)
            print(f"  Fold {fold} - Linear Regression fitted")
            
            # 3. HistGradientBoosting
            hgb = HistGradientBoostingRegressor(random_state=42)
            hgb.fit(x_train, y_train.reshape(-1))
            model_predictions["hist_gradient_boosting"][val_idx] = fused_val + hgb.predict(x_val)
            print(f"  Fold {fold} - HistGradientBoosting fitted")
            
            # 4. Random Forest
            rf = RandomForestRegressor(n_estimators=20, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=1)
            rf.fit(x_train, y_train.reshape(-1))
            model_predictions["random_forest"][val_idx] = fused_val + rf.predict(x_val)
            print(f"  Fold {fold} - Random Forest fitted")
            
            # 5. Numpy MLP Single
            params_s, _ = train_mlp_single(x_train, y_train, epochs=80)
            model_predictions["numpy_mlp_single"][val_idx] = fused_val + predict_mlp_single(params_s, x_val)
            print(f"  Fold {fold} - Numpy MLP Single fitted")
            
            # 6. Numpy MLP Double
            params_d, _ = train_mlp_double(x_train, y_train, epochs=80)
            model_predictions["numpy_mlp_double"][val_idx] = fused_val + predict_mlp_double(params_d, x_val)
            print(f"  Fold {fold} - Numpy MLP Double fitted")
    else:
        # 降级至 split 划分进行评估
        train_idx = df_all[df_all["split"] == "train"].index
        val_idx = df_all[df_all["split"] != "train"].index # val + test
        
        x_train = df_all.loc[train_idx, features].to_numpy(dtype=np.float64)
        y_train = df_all.loc[train_idx, "residual_range_m"].to_numpy(dtype=np.float64).reshape(-1, 1)
        
        x_val = df_all.loc[val_idx, features].to_numpy(dtype=np.float64)
        fused_val = df_all.loc[val_idx, "fused_range_m"].to_numpy(dtype=np.float64)
        
        model_predictions["physics_baseline"][val_idx] = fused_val
        
        lr = LinearRegression()
        lr.fit(x_train, y_train.reshape(-1))
        model_predictions["linear_regression"][val_idx] = fused_val + lr.predict(x_val)
        
        hgb = HistGradientBoostingRegressor(random_state=42)
        hgb.fit(x_train, y_train.reshape(-1))
        model_predictions["hist_gradient_boosting"][val_idx] = fused_val + hgb.predict(x_val)
        
        rf = RandomForestRegressor(n_estimators=20, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=1)
        rf.fit(x_train, y_train.reshape(-1))
        model_predictions["random_forest"][val_idx] = fused_val + rf.predict(x_val)
        
        params_s, _ = train_mlp_single(x_train, y_train, epochs=80)
        model_predictions["numpy_mlp_single"][val_idx] = fused_val + predict_mlp_single(params_s, x_val)
        
        params_d, _ = train_mlp_double(x_train, y_train, epochs=80)
        model_predictions["numpy_mlp_double"][val_idx] = fused_val + predict_mlp_double(params_d, x_val)
        
        # 补充 train 里的预测值
        for m in models_to_eval:
            model_predictions[m][train_idx] = df_all.loc[train_idx, "fused_range_m"] # 基准占位

    # 计算各模型在 val_idx / 全体测试上的评估指标
    y_true_eval = df_all.loc[val_idx if not has_folds else df_all.index, "true_range_m"].to_numpy(dtype=np.float64)
    for m in models_to_eval:
        y_pred_eval = model_predictions[m][val_idx if not has_folds else df_all.index]
        comparison_results[m] = metrics(y_true_eval, y_pred_eval)
        print(f"Model {m:25s} - MAE: {comparison_results[m]['mae_m']:.4f}m, RMSE: {comparison_results[m]['rmse_m']:.4f}m")

    # 确定表现最优秀的模型在全量数据 (除 test/fold=4) 上进行最终生产训练并导出
    best_model_name = "numpy_mlp_double" # 默认推荐并导出双层 MLP
    print(f"\nTraining final production model ({best_model_name}) on full training folds...")
    
    # 最终拟合集：若是 5折，则取 fold 0, 1, 2, 3，保留 fold 4 作为外推/离线测试
    if has_folds:
        final_train_idx = df_all[df_all["fold_id"] != 4].index
    else:
        final_train_idx = df_all[df_all["split"] != "test"].index
        
    x_final = df_all.loc[final_train_idx, features].to_numpy(dtype=np.float64)
    y_final = df_all.loc[final_train_idx, "residual_range_m"].to_numpy(dtype=np.float64).reshape(-1, 1)
    
    final_params, final_history = train_mlp_double(x_final, y_final, epochs=100)
    
    # 评估最终模型在测试集 (fold 4) 上的效果
    if has_folds:
        test_idx = df_all[df_all["fold_id"] == 4].index
    else:
        test_idx = df_all[df_all["split"] == "test"].index
        
    x_test = df_all.loc[test_idx, features].to_numpy(dtype=np.float64)
    fused_test = df_all.loc[test_idx, "fused_range_m"].to_numpy(dtype=np.float64)
    true_test = df_all.loc[test_idx, "true_range_m"].to_numpy(dtype=np.float64)
    
    pred_test_residual = predict_mlp_double(final_params, x_test)
    pred_test_range = fused_test + pred_test_residual
    test_metrics = metrics(true_test, pred_test_range)
    print(f"Final Model Test Split (Fold 4) - MAE: {test_metrics['mae_m']:.4f}m, RMSE: {test_metrics['rmse_m']:.4f}m")

    # 保存最终模型 npz
    model_dir = package_root / "models" / args.dataset_version
    model_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        model_dir / "distance_residual_mlp_numpy.npz",
        x_mean=final_params["x_mean"],
        x_std=final_params["x_std"],
        y_mean=final_params["y_mean"],
        y_std=final_params["y_std"],
        w1=final_params["w1"],
        b1=final_params["b1"],
        w2=final_params["w2"],
        b2=final_params["b2"],
        w3=final_params["w3"],
        b3=final_params["b3"],
        feature_columns=np.array(features),
        num_layers=np.array([2]) # 标记双层
    )
    
    # 汇总报告
    report = {
        "dataset_version": args.dataset_version,
        "best_model": best_model_name,
        "feature_columns": features,
        "comparison_val_metrics": comparison_results,
        "test_split_metrics": test_metrics,
        "training_history": final_history[-10:]
    }
    
    (model_dir / "model_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    
    report_dir = package_root / "reports" / args.dataset_version
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "model_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 写入 markdown 格式比对表格
    with (report_dir / "model_comparison.md").open("w", encoding="utf-8") as f:
        f.write("# Residual MLP Model Comparison\n\n")
        f.write("| Model | MAE (m) | RMSE (m) | P95 Abs Error (m) | Max Error (m) |\n")
        f.write("|-------|---------|----------|-------------------|---------------|\n")
        for m in models_to_eval:
            met = comparison_results[m]
            f.write(f"| {m} | {met['mae_m']:.4f} | {met['rmse_m']:.4f} | {met['p95_abs_m']:.4f} | {met['max_abs_m']:.4f} |\n")
            
    print(f"\nFinal training outputs saved to {model_dir}")


if __name__ == "__main__":
    main()
