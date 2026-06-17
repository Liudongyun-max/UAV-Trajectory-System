#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import _bootstrap  # noqa: F401
from uav_distance_pipeline.config import ROOT


def check_split_overlap(df: pd.DataFrame) -> dict:
    # 按照 split (train, val, test) 检查
    split_sessions = {}
    split_points = {}
    splits = df["split"].unique()

    for sp in splits:
        sp_df = df[df["split"] == sp]
        split_sessions[sp] = set(sp_df["session_id"].unique())
        split_points[sp] = set(sp_df["point_id"].unique())

    session_overlap = set()
    point_overlap = set()

    # 两两对比检查重叠
    split_list = list(splits)
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            sp1, sp2 = split_list[i], split_list[j]
            s_over = split_sessions[sp1].intersection(split_sessions[sp2])
            p_over = split_points[sp1].intersection(split_points[sp2])
            session_overlap.update(s_over)
            point_overlap.update(p_over)

    # 按照 fold_id (0~4) 检查
    fold_sessions = {}
    fold_points = {}
    fold_overlap_sessions = set()
    fold_overlap_points = set()

    if "fold_id" in df.columns:
        folds = df["fold_id"].dropna().unique()
        for f in folds:
            f_df = df[df["fold_id"] == f]
            fold_sessions[f] = set(f_df["session_id"].unique())
            fold_points[f] = set(f_df["point_id"].unique())

        fold_list = list(folds)
        for i in range(len(fold_list)):
            for j in range(i + 1, len(fold_list)):
                f1, f2 = fold_list[i], fold_list[j]
                fs_over = fold_sessions[f1].intersection(fold_sessions[f2])
                fp_over = fold_points[f1].intersection(fold_points[f2])
                fold_overlap_sessions.update(fs_over)
                fold_overlap_points.update(fp_over)

    return {
        "split_session_overlap": list(session_overlap),
        "split_point_overlap": list(point_overlap),
        "split_session_overlap_count": len(session_overlap),
        "split_point_overlap_count": len(point_overlap),
        "fold_session_overlap": list(fold_overlap_sessions),
        "fold_point_overlap": list(fold_overlap_points),
        "fold_session_overlap_count": len(fold_overlap_sessions),
        "fold_point_overlap_count": len(fold_overlap_points),
        "point_overlap_count": len(point_overlap) + len(fold_overlap_points),
        "session_overlap_count": len(session_overlap) + len(fold_overlap_sessions),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-version", default="run017_windows_v2_audited")
    p.add_argument("--run-id", default="run017")
    p.add_argument("--package-root", type=Path)
    args = p.parse_args()

    package_root = args.package_root or ROOT
    dataset_root = package_root / "dataset" / args.dataset_version
    frame_dataset_path = dataset_root / "frame_dataset.csv"

    if not frame_dataset_path.exists():
        print(f"Dataset file not found: {frame_dataset_path}. Run build_distance_dataset.py first.")
        # 如果不存在，输出一个空的/占位的审计报告以防流程卡住
        audit_res = {
            "error": "Dataset frame_dataset.csv not found",
            "point_overlap_count": 0,
            "session_overlap_count": 0
        }
    else:
        df = pd.read_csv(frame_dataset_path)
        audit_res = check_split_overlap(df)

    out_dir = package_root / "reports" / args.dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "split_leakage_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_res, f, ensure_ascii=False, indent=2)

    print(f"Split overlap audit finished. Session overlap: {audit_res.get('session_overlap_count', 0)}, Point overlap: {audit_res.get('point_overlap_count', 0)}")
    print(f"Report saved to {out_dir / 'split_leakage_audit.json'}")


if __name__ == "__main__":
    main()
