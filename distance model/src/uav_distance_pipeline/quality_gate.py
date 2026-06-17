from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointMetrics:
    rtk_fix: bool
    nominal_distance_error_m: float
    true_range_std_m: float
    sync_p95_ms: float
    capture_duration_s: float
    valid_frames: int
    visible_ratio: float
    scene_profile_consistent: bool


def scene_profile_consistent(scene_cfg: dict, wind_speed_mean_mps: float) -> bool:
    target = scene_cfg["wind_speed_target_mps"]
    return float(target["min"]) <= float(wind_speed_mean_mps) <= float(target["max"])


def grade_point(metrics: PointMetrics) -> tuple[str, bool, str]:
    if (
        metrics.rtk_fix
        and metrics.nominal_distance_error_m <= 0.10
        and metrics.true_range_std_m <= 0.05
        and metrics.sync_p95_ms <= 5
        and 4.9 <= metrics.capture_duration_s <= 5.1
        and metrics.valid_frames >= 123
        and metrics.visible_ratio >= 0.98
        and metrics.scene_profile_consistent
    ):
        return "A", True, "distance_points"
    if (
        metrics.rtk_fix
        and metrics.nominal_distance_error_m <= 0.20
        and metrics.true_range_std_m <= 0.10
        and metrics.sync_p95_ms <= 10
        and 4.8 <= metrics.capture_duration_s <= 5.2
        and metrics.valid_frames >= 120
        and metrics.visible_ratio >= 0.95
        and metrics.scene_profile_consistent
    ):
        return "B", True, "distance_points"
    if metrics.valid_frames >= 1 and metrics.rtk_fix:
        return "C", False, "quarantine"
    return "D", False, "rejected"

