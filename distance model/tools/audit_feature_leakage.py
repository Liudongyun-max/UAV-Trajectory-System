#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401
from uav_distance_pipeline.config import ROOT


# 禁止作为特征的字段/关键字
BANNED_KEYWORDS = [
    "true_range",
    "residual_range",
    "residual",
    "nominal_distance",
    "distance_bin",
    "range_error",
    "coordinate_error",
    "run_id",
    "session_id",
    "point_id",
    "point_path",
    "source_video",
]


def audit_features(feature_columns: list[str]) -> tuple[bool, list[str]]:
    leaked = []
    for col in feature_columns:
        col_lower = col.lower()
        for banned in BANNED_KEYWORDS:
            if banned in col_lower:
                leaked.append(col)
                break
    return len(leaked) == 0, leaked


def audit_physics_provenance(df: pd.DataFrame) -> dict:
    # 检查 fused_range_m 是否由 true_range_m 直接反推（检测完美相关或恒定误差）
    if "fused_range_m" not in df.columns or "true_range_m" not in df.columns:
        return {"status": "error", "message": "missing fused_range_m or true_range_m in dataset"}

    fused = df["fused_range_m"].to_numpy(dtype=float)
    true_val = df["true_range_m"].to_numpy(dtype=float)

    error = fused - true_val
    mean_err = np.mean(error)
    std_err = np.std(error)
    min_err = np.min(np.abs(error))

    # 如果误差的标准差为 0 且均值为 0，或者误差的标准差极其微小，可能存在真值反推
    is_suspicious = False
    reason = "Normal physical estimation (noisy)"

    if std_err < 1e-7:
        is_suspicious = True
        reason = "Extremely low error standard deviation, likely direct truth back-propagation!"
    elif min_err == 0.0 and std_err < 1e-4:
        is_suspicious = True
        reason = "Zero error on multiple frames with very low variance"

    return {
        "status": "fail" if is_suspicious else "pass",
        "reason": reason,
        "mean_error_m": float(mean_err),
        "std_error_m": float(std_err),
        "min_abs_error_m": float(min_err),
        "correlation": float(np.corrcoef(fused, true_val)[0, 1]) if len(fused) > 1 else 1.0
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-version", default="run017_windows_v2_audited")
    p.add_argument("--run-id", default="run017")
    p.add_argument("--package-root", type=Path)
    args = p.parse_args()

    package_root = args.package_root or ROOT
    dataset_root = package_root / "dataset" / args.dataset_version
    schema_path = dataset_root / "feature_schema.json"
    frame_dataset_path = dataset_root / "frame_dataset.csv"

    feature_leakage_res = {
        "status": "pass",
        "leaked_features": [],
        "banned_keywords_checked": BANNED_KEYWORDS
    }
    physics_prov_res = {
        "status": "unknown",
        "reason": "frame_dataset.csv not found"
    }

    if not schema_path.exists():
        print(f"feature_schema.json not found: {schema_path}")
        feature_leakage_res["status"] = "error"
        feature_leakage_res["message"] = "feature_schema.json not found"
    else:
        with schema_path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
        features = schema.get("feature_columns", [])
        is_clean, leaked = audit_features(features)
        if not is_clean:
            feature_leakage_res["status"] = "fail"
            feature_leakage_res["leaked_features"] = leaked
            print(f"Warning: Leaked features found: {leaked}")
        else:
            print("Feature columns audit passed (no banned fields).")

    if not frame_dataset_path.exists():
        print(f"frame_dataset.csv not found: {frame_dataset_path}")
    else:
        df = pd.read_csv(frame_dataset_path)
        physics_prov_res = audit_physics_provenance(df)
        print(f"Physics provenance check: {physics_prov_res['status']} ({physics_prov_res['reason']})")

    out_dir = package_root / "reports" / args.dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "feature_leakage_audit.json").open("w", encoding="utf-8") as f:
        json.dump(feature_leakage_res, f, ensure_ascii=False, indent=2)

    with (out_dir / "physics_feature_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(physics_prov_res, f, ensure_ascii=False, indent=2)

    print(f"Audits saved to {out_dir}")


if __name__ == "__main__":
    main()
