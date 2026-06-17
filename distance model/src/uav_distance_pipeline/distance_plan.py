from __future__ import annotations

from dataclasses import dataclass

from .config import expected_frames


@dataclass(frozen=True)
class SessionPlan:
    scene_profile: str
    distance_start_m: int
    distance_end_m: int
    session_index: int

    @property
    def session_id(self) -> str:
        return f"session_{self.scene_profile}_d{self.distance_start_m:03d}-d{self.distance_end_m:03d}_{self.session_index:03d}"

    @property
    def point_count(self) -> int:
        return self.distance_end_m - self.distance_start_m + 1


def distances(start_m: int, end_m: int, step_m: int) -> list[int]:
    if step_m <= 0:
        raise ValueError("distance step must be positive")
    if start_m > end_m:
        raise ValueError("distance start must be <= end")
    return list(range(start_m, end_m + 1, step_m))


def intersect_range(start_m: int, end_m: int, range_start: int, range_end: int) -> tuple[int, int] | None:
    lo = max(start_m, range_start)
    hi = min(end_m, range_end)
    if lo > hi:
        return None
    return lo, hi


def build_sessions(start_m: int, end_m: int, scenes: list[str]) -> list[SessionPlan]:
    standard_ranges = [(20, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    sessions: list[SessionPlan] = []
    for scene in scenes:
        idx = 1
        for range_start, range_end in standard_ranges:
            clipped = intersect_range(start_m, end_m, range_start, range_end)
            if clipped is None:
                continue
            sessions.append(SessionPlan(scene, clipped[0], clipped[1], idx))
            idx += 1
    return sessions


def plan_summary(
    run_id: str,
    requested_run_id: str,
    start_m: int,
    end_m: int,
    step_m: int,
    scenes: list[str],
    fps: float,
    capture_duration_s: float,
) -> dict:
    dists = distances(start_m, end_m, step_m)
    sessions = build_sessions(start_m, end_m, scenes)
    frames_per_point = expected_frames(fps, capture_duration_s)
    logical_points = len(dists) * len(scenes)
    return {
        "requested_run_id": requested_run_id,
        "resolved_run_id": run_id,
        "distance_start_m": start_m,
        "distance_end_m": end_m,
        "distance_step_m": step_m,
        "distance_points": len(dists),
        "scene_profiles": len(scenes),
        "scene_profile_names": scenes,
        "logical_points": logical_points,
        "fps": fps,
        "point_capture_duration_s": capture_duration_s,
        "expected_frames_per_point": frames_per_point,
        "expected_total_frames": logical_points * frames_per_point,
        "planned_sessions": len(sessions),
        "sessions": [
            {
                "session_id": session.session_id,
                "scene_profile": session.scene_profile,
                "distance_start_m": session.distance_start_m,
                "distance_end_m": session.distance_end_m,
                "point_count": session.point_count,
            }
            for session in sessions
        ],
    }

