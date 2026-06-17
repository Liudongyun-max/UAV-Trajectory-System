#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from uav_distance_pipeline.config import ROOT


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_point_integrity(point_dir: Path) -> tuple[bool, str, dict]:
    required_files = ["detections.csv", "truth.csv", "distance_estimates.csv", "point.yaml"]
    missing = [f for f in required_files if not (point_dir / f).exists()]
    if missing:
        return False, f"missing_files: {', '.join(missing)}", {}

    # 读取 point.yaml 确认属性
    point_yaml_path = point_dir / "point.yaml"
    try:
        import yaml
        with point_yaml_path.open("r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
    except Exception as e:
        return False, f"yaml_parse_error: {str(e)}", {}

    grade = meta.get("quality_grade", "D")
    valid_frames = meta.get("valid_frames", 0)

    if grade == "D":
        return False, "quality_grade_D", meta
    if valid_frames < 120:
        return False, f"low_valid_frames: {valid_frames}", meta

    # 读取 csv 检查行数
    dets = read_csv(point_dir / "detections.csv")
    truth = read_csv(point_dir / "truth.csv")
    est = read_csv(point_dir / "distance_estimates.csv")

    if not dets:
        return False, "empty_detections", meta
    if not truth:
        return False, "empty_truth", meta
    if not est:
        return False, "empty_estimates", meta

    return True, "", meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="run017")
    p.add_argument("--package-root", type=Path)
    args = p.parse_args()

    package_root = args.package_root or ROOT
    manifest_path = package_root / "input" / "manifests" / "point_manifest_windows.csv"
    if not manifest_path.exists():
        manifest_path = package_root / "input" / "manifests" / "point_manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found, scanning directory instead: {package_root / 'input' / 'distance_points' / args.run_id}")
        run_dir = package_root / "input" / "distance_points" / args.run_id
        points = []
        for p_dir in run_dir.glob("**/point.yaml"):
            points.append({
                "point_id": p_dir.parent.name,
                "point_path_relative": str(p_dir.parent.relative_to(package_root)),
                "nominal_distance_m": "0",
                "scene_profile": p_dir.parent.parent.parent.name,
                "quality_grade": "A",
                "valid_frames": "125"
            })
    else:
        points = read_csv(manifest_path)

    integrity_report = {
        "run_id": args.run_id,
        "total_manifest_points": len(points),
        "valid_points_count": 0,
        "skipped_points_count": 0,
        "points": {}
    }

    skipped_rows = []
    skipped_fields = ["point_id", "session_id", "nominal_distance_m", "quality_grade", "valid_frames", "skip_reason"]

    for pt in points:
        pt_id = pt.get("point_id", "")
        pt_rel = pt.get("point_path_relative", "")
        if pt_rel:
            pt_dir = package_root / pt_rel
        else:
            pt_dir = Path(pt.get("point_path", ""))

        is_valid, reason, meta = check_point_integrity(pt_dir)
        pt_session = pt.get("session_id", meta.get("session_id", "unknown"))
        pt_grade = pt.get("quality_grade", meta.get("quality_grade", "D"))
        pt_valid_frames = pt.get("valid_frames", meta.get("valid_frames", 0))
        pt_dist = pt.get("nominal_distance_m", meta.get("nominal_distance_m", 0))

        if is_valid:
            integrity_report["valid_points_count"] += 1
            integrity_report["points"][pt_id] = {
                "status": "valid",
                "session_id": pt_session,
                "path": str(pt_rel)
            }
        else:
            integrity_report["skipped_points_count"] += 1
            integrity_report["points"][pt_id] = {
                "status": "skipped",
                "reason": reason,
                "session_id": pt_session,
                "path": str(pt_rel)
            }
            skipped_rows.append({
                "point_id": pt_id,
                "session_id": pt_session,
                "nominal_distance_m": pt_dist,
                "quality_grade": pt_grade,
                "valid_frames": pt_valid_frames,
                "skip_reason": reason
            })

    # 写入报告
    out_dir = package_root / "reports" / f"{args.run_id}_windows_v2_audited"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "package_integrity.json").open("w", encoding="utf-8") as f:
        json.dump(integrity_report, f, ensure_ascii=False, indent=2)

    with (out_dir / "skipped_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=skipped_fields)
        writer.writeheader()
        writer.writerows(skipped_rows)

    print(f"Audited {len(points)} points. Valid: {integrity_report['valid_points_count']}, Skipped: {integrity_report['skipped_points_count']}")
    print(f"Reports saved to {out_dir}")


if __name__ == "__main__":
    main()
