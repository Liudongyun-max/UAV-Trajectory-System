#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401
from uav_distance_pipeline.config import ROOT


def load_mlp_model(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"MLP model npz not found: {model_path}")
    data = np.load(model_path, allow_pickle=True)
    params = {
        "x_mean": data["x_mean"],
        "x_std": data["x_std"],
        "y_mean": data["y_mean"],
        "y_std": data["y_std"],
        "w1": data["w1"],
        "b1": data["b1"],
        "w2": data["w2"],
        "b2": data["b2"],
        "feature_columns": list(data["feature_columns"]),
    }
    if "w3" in data:
        params["w3"] = data["w3"]
        params["b3"] = data["b3"]
        params["num_layers"] = 2
    else:
        params["num_layers"] = 1
    return params


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def predict_mlp(params: dict, x: np.ndarray) -> np.ndarray:
    xs = (x - params["x_mean"]) / params["x_std"]
    if params["num_layers"] == 2:
        h1 = relu(xs @ params["w1"] + params["b1"])
        h2 = relu(h1 @ params["w2"] + params["b2"])
        pred_scaled = h2 @ params["w3"] + params["b3"]
    else:
        h = relu(xs @ params["w1"] + params["b1"])
        pred_scaled = h @ params["w2"] + params["b2"]
    return (pred_scaled * params["y_std"] + params["y_mean"]).reshape(-1)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="run017")
    p.add_argument("--dataset-version", default="run017_windows_v2_audited")
    p.add_argument("--model-version", default="run017_windows_v2_residual")
    p.add_argument("--package-root", type=Path)
    args = p.parse_args()

    package_root = args.package_root or ROOT
    dataset_root = package_root / "dataset" / args.dataset_version
    model_path = package_root / "models" / args.model_version / "distance_residual_mlp_numpy.npz"

    # 加载 MLP
    mlp_params = load_mlp_model(model_path)
    feature_cols = mlp_params["feature_columns"]

    # 读取 manifests
    manifest_path = package_root / "input" / "manifests" / "point_manifest_windows.csv"
    if not manifest_path.exists():
        manifest_path = package_root / "input" / "manifests" / "point_manifest.csv"
    
    point_rows = read_csv(manifest_path)
    
    # 获取 Session fold 分配
    fold_assignments_path = dataset_root / "fold_assignments.csv"
    if fold_assignments_path.exists():
        fold_df = pd.read_csv(fold_assignments_path)
        session_to_fold = dict(zip(fold_df["session_id"], fold_df["fold_id"]))
    else:
        session_to_fold = {row["session_id"]: 0 for row in point_rows} # 降级

    all_frames_data = []

    # 数值特征与时间差对齐
    numeric_features = [
        "bbox_width_px", "bbox_height_px", "bbox_area_px", "bbox_center_u", "bbox_center_v",
        "detector_confidence", "physics_width_range_m", "physics_height_range_m",
        "physics_area_range_m", "fused_range_m", "bbox_aspect_ratio",
        "bbox_center_u_norm", "bbox_center_v_norm"
    ]

    print(f"Aligning time series data and predicting residual for {len(point_rows)} points...")

    for pt in point_rows:
        grade = pt.get("quality_grade", "")
        valid_frames = int(float(pt.get("valid_frames") or 0))
        if grade == "D" or valid_frames < 120:
            continue

        point_id = pt["point_id"]
        session_id = pt["session_id"]
        fold_id = session_to_fold.get(session_id, 0)
        scene = pt["scene_profile"]

        point_rel = pt.get("point_path_relative")
        if point_rel:
            point_dir = package_root / point_rel
        else:
            point_dir = Path(pt["point_path"])

        # 读取 truth, detections, estimates
        truth_df = pd.read_csv(point_dir / "truth.csv")
        # 确保 truth 按 frame_id 排序
        truth_df = truth_df.sort_values("frame_id").reset_index(drop=True)

        det_df = pd.read_csv(point_dir / "detections.csv")
        est_df = pd.read_csv(point_dir / "distance_estimates.csv")

        # 派生特征
        det_df["bbox_aspect_ratio"] = det_df["bbox_width_px"] / det_df["bbox_height_px"]
        det_df["bbox_center_u_norm"] = det_df["bbox_center_u"] / 2560.0
        det_df["bbox_center_v_norm"] = det_df["bbox_center_v"] / 1440.0

        # 以 truth 为基表 merge，对齐时间序列
        det_df_clean = det_df.drop(columns=["true_range_m", "timestamp"], errors="ignore")
        est_df_clean = est_df.drop(columns=["true_range_m", "timestamp"], errors="ignore")
        
        merged = pd.merge(truth_df[["frame_id", "timestamp", "true_range_m"]], det_df_clean, on="frame_id", how="left")
        merged = pd.merge(merged, est_df_clean, on="frame_id", how="left")

        # 标记缺失帧与插值帧
        merged["is_missing"] = merged["bbox_width_px"].isna().astype(int)
        
        # 填充缺失的数值特征 (插值对齐)
        merged[numeric_features] = merged[numeric_features].interpolate(method="linear")
        # 首尾如果有 NaN，使用 ffill/bfill 兜底
        merged[numeric_features] = merged[numeric_features].ffill().bfill()

        # 对填充过的行设为 interpolated
        merged["is_interpolated"] = (merged["is_missing"] & ~merged["bbox_width_px"].isna().astype(int)).astype(int)
        # 刚才填充之后，原本是 nan 的现在不再是 nan，所以 interpolated 应为 is_missing 为 1 的那些
        merged["is_interpolated"] = merged["is_missing"] # 只要本来缺失被补齐了，就是插值

        # 补充非数值字段
        merged["run_id"] = args.run_id
        merged["session_id"] = session_id
        merged["point_id"] = point_id
        merged["fold_id"] = fold_id
        merged["scene_profile"] = scene
        merged["nominal_distance_m"] = float(pt.get("nominal_distance_m", 0.0))
        merged["distance_bin_start_m"] = float(pt.get("distance_bin_start_m", 0.0))
        merged["distance_bin_end_m"] = float(pt.get("distance_bin_end_m", 0.0))

        # 计算 frame_delta_t_s
        merged["frame_delta_t_s"] = merged["timestamp"].diff()
        merged["frame_delta_t_s"] = merged["frame_delta_t_s"].fillna(0.04) # 25 FPS 默认 0.04s

        # MLP 预测误差残差
        x_mlp = merged[feature_cols].to_numpy(dtype=np.float64)
        pred_res = predict_mlp(mlp_params, x_mlp)
        
        merged["mlp_predicted_residual_m"] = pred_res
        merged["raw_predicted_range_m"] = merged["fused_range_m"] + pred_res

        all_frames_data.append(merged)

    # 汇总
    full_df = pd.concat(all_frames_data, ignore_index=True)

    # 创建输出目录
    out_dir = package_root / "exports" / "gru_input" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 输出 frame_predictions.csv (Parquet fallback)
    predictions_csv_path = out_dir / "frame_predictions.csv"
    full_df.to_csv(predictions_csv_path, index=False)
    print(f"Exported frame_predictions to {predictions_csv_path}")

    # 复制 manifests 形成时序交付包
    pd.read_csv(manifest_path).to_csv(out_dir / "point_manifest.csv", index=False)
    
    session_manifest_path = package_root / "input" / "manifests" / "session_manifest_windows.csv"
    if not session_manifest_path.exists():
        session_manifest_path = package_root / "input" / "manifests" / "session_manifest.csv"
    pd.read_csv(session_manifest_path).to_csv(out_dir / "session_manifest.csv", index=False)

    if fold_assignments_path.exists():
        pd.read_csv(fold_assignments_path).to_csv(out_dir / "fold_assignments.csv", index=False)

    # 提取并计算标准化参数 (为 GRU 输入进行 Normalization)
    gru_feature_cols = [
        "bbox_width_px", "bbox_height_px", "bbox_area_px", "bbox_aspect_ratio",
        "bbox_center_u_norm", "bbox_center_v_norm", "detector_confidence",
        "is_missing", "is_interpolated", "physics_width_range_m", "physics_height_range_m",
        "physics_area_range_m", "fused_range_m", "mlp_predicted_residual_m",
        "raw_predicted_range_m", "frame_delta_t_s"
    ]

    mean_vals = full_df[gru_feature_cols].mean().to_dict()
    std_vals = full_df[gru_feature_cols].std().to_dict()
    # 避免 std 为 0
    for k in std_vals:
        if std_vals[k] < 1e-9:
            std_vals[k] = 1.0

    norm_info = {"mean": mean_vals, "std": std_vals}
    with (out_dir / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump(norm_info, f, ensure_ascii=False, indent=2)

    # 输出 feature_schema.json
    gru_schema = {
        "feature_columns": gru_feature_cols,
        "target_columns": ["true_range_m"],
        "primary_target": "true_range_m",
        "input_dim": len(gru_feature_cols),
        "sequence_length": 25,
        "stride": 5
    }
    with (out_dir / "feature_schema.json").open("w", encoding="utf-8") as f:
        json.dump(gru_schema, f, ensure_ascii=False, indent=2)

    # 输出 export_manifest.json
    export_manifest = {
        "dataset_version": args.dataset_version,
        "run_id": args.run_id,
        "export_time": pd.Timestamp.now().isoformat(),
        "total_points": len(all_frames_data),
        "total_frames": len(full_df),
        "parquet_fallback": True,
        "features": gru_feature_cols,
    }
    with (out_dir / "export_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(export_manifest, f, ensure_ascii=False, indent=2)

    print(f"GRU input delivery package generated successfully at {out_dir}")


if __name__ == "__main__":
    main()
