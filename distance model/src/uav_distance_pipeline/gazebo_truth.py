from __future__ import annotations

import math


def true_range_m(xc: float, yc: float, zc: float) -> float:
    return math.sqrt(xc * xc + yc * yc + zc * zc)


def target_position_from_camera(nominal_distance_m: float, jitter_y_m: float = 0.0, jitter_z_m: float = 0.0) -> tuple[float, float, float]:
    x = float(nominal_distance_m)
    y = float(jitter_y_m)
    z = float(jitter_z_m)
    return x, y, z


def stable_motion(frame_id: int, fps: float) -> tuple[float, float, float, float, float]:
    t = frame_id / fps
    roll_deg = 0.2 * math.sin(t * 0.7)
    pitch_deg = 0.15 * math.cos(t * 0.5)
    yaw_deg = 0.3 * math.sin(t * 0.3)
    horizontal_speed_mps = 0.02
    vertical_speed_mps = 0.01
    return roll_deg, pitch_deg, yaw_deg, horizontal_speed_mps, vertical_speed_mps

