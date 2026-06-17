#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from .temporal_resampler import TemporalResampler


class DistanceSequenceBuilder:
    @staticmethod
    def build_sequences(
        source_root: Path,
        sequence_length: int = 25,
        stride: int = 5,
        scene_features: bool = False
    ) -> dict:
        """
        读取交付数据包, 构建滑动窗口时序序列。
        支持 25 FPS 和 15 FPS 等效视图重采样。
        """
        preds_path = source_root / "frame_predictions.csv"
        schema_path = source_root / "feature_schema.json"
        fold_path = source_root / "fold_assignments.csv"
        
        if not preds_path.exists():
            raise FileNotFoundError(f"Missing frame_predictions: {preds_path}")
            
        df_all = pd.read_csv(preds_path)
        with schema_path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
            
        feature_cols = schema["feature_columns"]
        
        # 加载归一化均值标准差
        norm_path = source_root / "normalization.json"
        with norm_path.open("r", encoding="utf-8") as f:
            norm_info = json.load(f)
        means = np.array([norm_info["mean"][c] for c in feature_cols], dtype=np.float32)
        stds = np.array([norm_info["std"][c] for c in feature_cols], dtype=np.float32)

        # 5折数据划分容器
        # 每个 fold 包含: 'x' (B, L, D), 'y' (B, 1), 'meta' (B, ...)
        fold_data = {f_id: {"x": [], "y": [], "missing_mask": [], "interpolated_mask": [], "meta": []} for f_id in range(5)}

        # 按 Point 分组, 确保窗口不跨 Point、Session 和 Split 边界
        grouped = df_all.groupby("point_id", sort=False)
        
        print(f"Building sliding window sequences (len={sequence_length}, stride={stride}) for {len(grouped)} points...")

        for point_id, df_pt in grouped:
            # 确保按 frame_id 递增
            df_pt = df_pt.sort_values("frame_id").reset_index(drop=True)
            fold_id = int(df_pt["fold_id"].iloc[0])
            scene = df_pt["scene_profile"].iloc[0]
            
            # --- 1. 25 FPS 原始序列视图 ---
            n_frames = len(df_pt)
            for start in range(0, n_frames - sequence_length + 1, stride):
                end = start + sequence_length
                sub_df = df_pt.iloc[start:end]
                
                # 均值/标准差归一化
                x_raw = sub_df[feature_cols].to_numpy(dtype=np.float32)
                x_norm = (x_raw - means) / stds
                
                # 最后一帧 true_range_m 作为 label
                y_label = float(sub_df["true_range_m"].iloc[-1])
                
                missing = sub_df["is_missing"].to_numpy(dtype=np.float32)
                interpolated = sub_df["is_interpolated"].to_numpy(dtype=np.float32)
                
                fold_data[fold_id]["x"].append(x_norm)
                fold_data[fold_id]["y"].append([y_label])
                fold_data[fold_id]["missing_mask"].append(missing)
                fold_data[fold_id]["interpolated_mask"].append(interpolated)
                
                fold_data[fold_id]["meta"].append({
                    "point_id": point_id,
                    "session_id": df_pt["session_id"].iloc[0],
                    "scene_profile": scene,
                    "nominal_distance": float(df_pt["nominal_distance_m"].iloc[-1]),
                    "effective_fps": 25.0,
                    "raw_preds": sub_df["raw_predicted_range_m"].to_numpy(dtype=np.float32).tolist(),
                    "is_missing": sub_df["is_missing"].to_numpy(dtype=np.float32).tolist(),
                })
                
            # --- 2. 15 FPS 等效重采样视图 ---
            # 同样以 stride 滑动作为起始帧进行 15 FPS 重采样
            for start in range(0, n_frames, stride):
                sub_df_15 = TemporalResampler.resample_to_15fps(df_pt, start, sequence_length)
                if sub_df_15 is None:
                    # 剩余帧数时间跨度不够
                    break
                    
                x_raw_15 = sub_df_15[feature_cols].to_numpy(dtype=np.float32)
                x_norm_15 = (x_raw_15 - means) / stds
                
                y_label_15 = float(sub_df_15["true_range_m"].iloc[-1])
                
                missing_15 = sub_df_15["is_missing"].to_numpy(dtype=np.float32)
                interpolated_15 = sub_df_15["is_interpolated"].to_numpy(dtype=np.float32)
                
                fold_data[fold_id]["x"].append(x_norm_15)
                fold_data[fold_id]["y"].append([y_label_15])
                fold_data[fold_id]["missing_mask"].append(missing_15)
                fold_data[fold_id]["interpolated_mask"].append(interpolated_15)
                
                fold_data[fold_id]["meta"].append({
                    "point_id": point_id,
                    "session_id": df_pt["session_id"].iloc[0],
                    "scene_profile": scene,
                    "nominal_distance": float(df_pt["nominal_distance_m"].iloc[-1]),
                    "effective_fps": 15.0,
                    "raw_preds": sub_df_15["raw_predicted_range_m"].to_numpy(dtype=np.float32).tolist(),
                    "is_missing": sub_df_15["is_missing"].to_numpy(dtype=np.float32).tolist(),
                })

        # 将数据转换为 numpy 数组
        final_folds = {}
        for f_id in range(5):
            if len(fold_data[f_id]["x"]) > 0:
                final_folds[f_id] = {
                    "x": np.array(fold_data[f_id]["x"], dtype=np.float32),
                    "y": np.array(fold_data[f_id]["y"], dtype=np.float32),
                    "missing_mask": np.array(fold_data[f_id]["missing_mask"], dtype=np.float32),
                    "interpolated_mask": np.array(fold_data[f_id]["interpolated_mask"], dtype=np.float32),
                    "meta": fold_data[f_id]["meta"]
                }
                print(f"Fold {f_id} built: {len(final_folds[f_id]['x'])} sequences.")
            else:
                final_folds[f_id] = None
                
        return final_folds
