from __future__ import annotations

from pathlib import Path

from .config import ROOT


RUN_SCAN_DIRS = (
    "data/raw_sessions",
    "data/distance_points",
    "data/quarantine",
    "data/rejected",
    "data/manifests",
    "analysis/runs",
)


def ensure_base_dirs(root: Path = ROOT) -> None:
    dirs = [
        "configs/camera_profiles",
        "configs/target_profiles",
        "configs/scene_profiles",
        "configs/training",
        "calibration/intrinsic",
        "calibration/extrinsic",
        "calibration/range_truth",
        "data/raw_sessions",
        "data/distance_points",
        "data/quarantine",
        "data/rejected",
        "data/recollection_queue",
        "data/manifests",
        "processed/aligned",
        "processed/detections",
        "processed/features",
        "processed/distance_labels",
        "processed/indexes",
        "datasets/distance_regression",
        "analysis/runs",
        "analysis/scenes",
        "analysis/distances",
        "analysis/datasets",
        "analysis/comparisons",
        "models/baselines",
        "models/trained",
        "outputs/predictions",
        "outputs/benchmarks",
        "outputs/figures",
        "outputs/deployment",
        "exports/windows",
        "exports/model_training",
        "exports/deployment",
        "logs",
        "cache",
        "tmp",
    ]
    for rel in dirs:
        (root / rel).mkdir(parents=True, exist_ok=True)


def run_dir(root: Path, area: str, run_id: str) -> Path:
    return root / area / run_id


def raw_session_path(root: Path, run_id: str, scene: str, session_id: str) -> Path:
    return root / "data" / "raw_sessions" / run_id / scene / session_id


def point_path(root: Path, run_id: str, scene: str, distance_m: int, point_id: str) -> Path:
    return root / "data" / "distance_points" / run_id / scene / f"d{distance_m:03d}" / point_id

