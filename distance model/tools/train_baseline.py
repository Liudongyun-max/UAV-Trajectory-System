#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from uav_distance_pipeline.config import ROOT


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_pred - y_true)
    return {
        "count": int(len(err)),
        "mae_m": float(np.mean(err)),
        "rmse_m": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "p50_abs_m": float(np.percentile(err, 50)),
        "p90_abs_m": float(np.percentile(err, 90)),
        "p95_abs_m": float(np.percentile(err, 95)),
        "max_abs_m": float(np.max(err)),
    }


def evaluate(dataset_root: Path) -> dict:
    report = {"dataset_root": str(dataset_root), "model": "physics_fused_baseline", "splits": {}, "scene_metrics": {}}
    for split in ("train", "val", "test"):
        path = dataset_root / f"{split}.csv"
        df = pd.read_csv(path)
        y_true = df["true_range_m"].to_numpy(dtype=float)
        y_pred = df["fused_range_m"].to_numpy(dtype=float)
        report["splits"][split] = metrics(y_true, y_pred)
    all_df = pd.read_csv(dataset_root / "frame_dataset.csv")
    for scene, df in all_df.groupby("scene_profile"):
        report["scene_metrics"][scene] = metrics(
            df["true_range_m"].to_numpy(dtype=float),
            df["fused_range_m"].to_numpy(dtype=float),
        )
    bucket_metrics = {}
    all_df["distance_bucket"] = (all_df["nominal_distance_m"] // 50 * 50).astype(int)
    for bucket, df in all_df.groupby("distance_bucket"):
        bucket_metrics[f"{bucket:03d}-{bucket + 49:03d}"] = metrics(
            df["true_range_m"].to_numpy(dtype=float),
            df["fused_range_m"].to_numpy(dtype=float),
        )
    report["distance_bucket_metrics"] = bucket_metrics
    return report


def default_package_root(run_id: str) -> Path:
    if (ROOT / "input" / "manifests").exists():
        return ROOT
    return ROOT / "exports" / "windows" / run_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--run-id", default="run017")
    p.add_argument("--package-root", type=Path)
    args = p.parse_args()
    package_root = args.package_root or default_package_root(args.run_id)
    dataset_root = package_root / "dataset" / args.dataset_version
    report = evaluate(dataset_root)
    out = package_root / "reports" / args.dataset_version
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
