from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraProfile:
    width: int
    height: int
    fps: float
    fx: float
    fy: float
    cx: float
    cy: float
    native_width: int
    native_height: int
    scaling_ratio: float


def default_camera_profile() -> CameraProfile:
    return CameraProfile(
        width=2560,
        height=1440,
        fps=25.0,
        fx=11494.25,
        fy=11494.25,
        cx=1280.0,
        cy=720.0,
        native_width=3840,
        native_height=2160,
        scaling_ratio=0.6666667,
    )


def project_point(xc: float, yc: float, zc: float, camera: CameraProfile | None = None) -> tuple[float, float]:
    cam = camera or default_camera_profile()
    if xc <= 0:
        return float("nan"), float("nan")
    u = cam.cx + cam.fx * (yc / xc)
    v = cam.cy - cam.fy * (zc / xc)
    return u, v


def back_project(u: float, v: float, range_m: float, camera: CameraProfile | None = None) -> tuple[float, float, float]:
    cam = camera or default_camera_profile()
    x = float(range_m)
    y = (u - cam.cx) * x / cam.fx
    z = -(v - cam.cy) * x / cam.fy
    return x, y, z


def bbox_from_target(range_m: float, target_width_m: float = 0.30, target_height_m: float = 0.15, camera: CameraProfile | None = None) -> tuple[float, float]:
    cam = camera or default_camera_profile()
    bbox_w = cam.fx * target_width_m / range_m
    bbox_h = cam.fy * target_height_m / range_m
    return max(1.0, bbox_w), max(1.0, bbox_h)

