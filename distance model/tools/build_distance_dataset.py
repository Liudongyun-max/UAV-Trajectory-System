#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from uav_distance_pipeline.config import ROOT

# 基础特征列 (Scene-Agnostic 必选与派生特征)
BASE_FEATURE_COLUMNS = [
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px",
    "bbox_center_u",
    "bbox_center_v",
    "detector_confidence",
    "physics_width_range_m",
    "physics_height_range_m",
    "physics_area_range_m",
    "fused_range_m",
    "bbox_aspect_ratio",
    "bbox_center_u_norm",
    "bbox_center_v_norm",
]

SCENE_FEATURE_COLUMNS = [
    "scene_empty_clean_no_wind",
    "scene_flat_ground_light_wind",
    "scene_forest_edge_moderate_wind",
]

TARGET_COLUMNS = [
    "true_range_m",
    "residual_range_m",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def split_name_legacy(distance: int, scene: str) -> str:
    if distance >= 400:
        return "test"
    if distance % 10 == 0 or scene == "forest_edge_moderate_wind" and distance % 25 == 0:
        return "val"
    return "train"


def build_dataset(
    package_root: Path,
    run_id: str,
    dataset_version: str,
    split_protocol: str,
    scene_features: bool
) -> dict:
    manifest = package_root / "input" / "manifests" / "point_manifest_windows.csv"
    if not manifest.exists():
        manifest = package_root / "input" / "manifests" / "point_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing point manifest: {manifest}")

    point_rows = read_csv(manifest)
    dataset_root = package_root / "dataset" / dataset_version
    if dataset_root.exists():
        import shutil
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True)

    # 1. 过滤不合规的 points 并确定 Session 集合
    active_points = []
    skipped_points = []
    for point in point_rows:
        grade = point.get("quality_grade", "")
        valid_frames = int(float(point.get("valid_frames") or 0))
        if grade == "D" or valid_frames < 120:
            skipped_points.append(point.get("point_id", ""))
            continue
        active_points.append(point)

    # 2. 数据划分协议分配
    session_to_fold = {}
    if split_protocol == "session-group-kfold":
        # 收集所有唯一的 session_id 并排序以保证确定性
        unique_sessions = sorted(list(set(pt["session_id"] for pt in active_points)))
        # 均匀分配至 5 个 fold
        for i, sid in enumerate(unique_sessions):
            session_to_fold[sid] = i % 5
    else:
        # legacy 协议下 fold 设为 -1
        for pt in active_points:
            session_to_fold[pt["session_id"]] = -1

    # 确定特征列
    feature_cols = list(BASE_FEATURE_COLUMNS)
    if scene_features:
        feature_cols += SCENE_FEATURE_COLUMNS

    frame_rows: list[dict] = []
    for point in active_points:
        point_rel = point.get("point_path_relative")
        if point_rel:
            point_dir = package_root / point_rel
        else:
            point_dir = Path(point["point_path"])

        detections = read_csv(point_dir / "detections.csv")
        truth = {int(row["frame_id"]): row for row in read_csv(point_dir / "truth.csv")}
        estimates = {int(row["frame_id"]): row for row in read_csv(point_dir / "distance_estimates.csv")}
        
        scene = point["scene_profile"]
        distance = int(float(point["nominal_distance_m"]))
        sid = point["session_id"]
        fold_id = session_to_fold[sid]

        # 确定 split 属性
        if split_protocol == "session-group-kfold":
            # fold 0, 1, 2 为 train, 3 为 val, 4 为 test
            if fold_id in (0, 1, 2):
                split = "train"
            elif fold_id == 3:
                split = "val"
            else:
                split = "test"
        else:
            split = split_name_legacy(distance, scene)

        for det in detections:
            frame_id = int(det["frame_id"])
            true_row = truth.get(frame_id)
            est_row = estimates.get(frame_id)
            if not true_row or not est_row:
                continue

            fused = as_float(est_row, "fused_range_m")
            true_range = as_float(true_row, "true_range_m")
            
            # 派生特征计算
            w = as_float(det, "bbox_width_px")
            h = as_float(det, "bbox_height_px")
            u = as_float(det, "bbox_center_u")
            v = as_float(det, "bbox_center_v")
            
            aspect_ratio = w / h if h > 0 else 0.0
            u_norm = u / 2560.0
            v_norm = v / 1440.0

            row = {
                "dataset_version": dataset_version,
                "split": split,
                "fold_id": fold_id,
                "run_id": run_id,
                "scene_profile": scene,
                "session_id": sid,
                "point_id": point["point_id"],
                "frame_id": frame_id,
                "nominal_distance_m": f"{distance:.6f}",
                "distance_bin_start_m": point["distance_bin_start_m"],
                "distance_bin_end_m": point["distance_bin_end_m"],
                "quality_grade": point.get("quality_grade", ""),
                "point_path_relative": point.get("point_path_relative", ""),
                "source_video_relative": point.get("source_video_relative", ""),
                
                # 必选特征
                "bbox_width_px": w,
                "bbox_height_px": h,
                "bbox_area_px": as_float(det, "bbox_area_px"),
                "bbox_center_u": u,
                "bbox_center_v": v,
                "detector_confidence": as_float(det, "detector_confidence"),
                "physics_width_range_m": est_row["physics_width_range_m"],
                "physics_height_range_m": est_row["physics_height_range_m"],
                "physics_area_range_m": est_row["physics_area_range_m"],
                "fused_range_m": fused,
                
                # 派生特征
                "bbox_aspect_ratio": aspect_ratio,
                "bbox_center_u_norm": u_norm,
                "bbox_center_v_norm": v_norm,
                
                # 标签
                "true_range_m": true_range,
                "residual_range_m": f"{true_range - fused:.9f}",
            }

            # 场景 One-Hot (无论是否用作 feature，均写在 CSV 中以支持评估，但 feature_schema.json 决定模型输入特征)
            row["scene_empty_clean_no_wind"] = 1 if scene == "empty_clean_no_wind" else 0
            row["scene_flat_ground_light_wind"] = 1 if scene == "flat_ground_light_wind" else 0
            row["scene_forest_edge_moderate_wind"] = 1 if scene == "forest_edge_moderate_wind" else 0

            frame_rows.append(row)

    base_columns = [
        "dataset_version", "split", "fold_id", "run_id", "scene_profile", "session_id", "point_id", "frame_id",
        "quality_grade", "point_path_relative", "source_video_relative",
    ]
    
    # 物理估计和场景元数据列
    meta_columns = [
        "nominal_distance_m", "distance_bin_start_m", "distance_bin_end_m",
        "scene_empty_clean_no_wind", "scene_flat_ground_light_wind", "scene_forest_edge_moderate_wind"
    ]
    
    fieldnames = base_columns + list(set(feature_cols + meta_columns)) + TARGET_COLUMNS
    write_csv(dataset_root / "frame_dataset.csv", frame_rows, fieldnames)
    
    split_counts = Counter(row["split"] for row in frame_rows)
    grade_counts = Counter(row["quality_grade"] for row in frame_rows)
    fold_counts = Counter(row["fold_id"] for row in frame_rows)

    for split in ("train", "val", "test"):
        split_rows = [row for row in frame_rows if row["split"] == split]
        write_csv(dataset_root / f"{split}.csv", split_rows, fieldnames)

    # 导出 feature_schema 供后续训练读取
    feature_schema = {
        "feature_columns": feature_cols,
        "target_columns": TARGET_COLUMNS,
        "primary_target": "residual_range_m",
        "final_prediction": "fused_range_m + residual_range_m",
        "label_source": "truth.csv true_range_m",
    }
    (dataset_root / "feature_schema.json").write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    # 导出 fold 分配表供 GRU 时序构建读取
    fold_assignments = []
    seen_sessions = set()
    for row in frame_rows:
        sid = row["session_id"]
        if sid not in seen_sessions:
            seen_sessions.add(sid)
            fold_assignments.append({
                "session_id": sid,
                "fold_id": row["fold_id"]
            })
    
    # 写入 fold_assignments.csv
    fold_fields = ["session_id", "fold_id"]
    write_csv(dataset_root / "fold_assignments.csv", fold_assignments, fold_fields)

    summary = {
        "dataset_version": dataset_version,
        "run_id": run_id,
        "package_root": str(package_root),
        "dataset_root": str(dataset_root),
        "point_count_total": len(point_rows),
        "point_count_skipped": len(skipped_points),
        "frame_count": len(frame_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "grade_counts": dict(sorted(grade_counts.items())),
        "fold_counts": dict(sorted(fold_counts.items())),
        "feature_columns": feature_cols,
        "target_columns": TARGET_COLUMNS,
    }
    (dataset_root / "dataset_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def default_package_root(run_id: str) -> Path:
    if (ROOT / "input" / "manifests").exists():
        return ROOT
    return ROOT / "exports" / "windows" / run_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--run-id", default="run017")
    p.add_argument("--package-root", type=Path)
    p.add_argument("--split-protocol", default="session-group-kfold", choices=["session-group-kfold", "legacy"])
    p.add_argument("--scene-features", default="false") # 默认为字符串, 后续转为布尔值
    args = p.parse_args()
    
    scene_features_bool = args.scene_features.lower() in ("true", "1", "yes")
    package_root = args.package_root or default_package_root(args.run_id)
    
    summary = build_dataset(
        package_root=package_root,
        run_id=args.run_id,
        dataset_version=args.dataset_version,
        split_protocol=args.split_protocol,
        scene_features=scene_features_bool
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
