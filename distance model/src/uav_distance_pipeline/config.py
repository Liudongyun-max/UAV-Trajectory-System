from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
VALID_SCENES = (
    "empty_clean_no_wind",
    "flat_ground_light_wind",
    "forest_edge_moderate_wind",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(name: str) -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / name)


def parse_scenes(value: str) -> list[str]:
    scenes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [scene for scene in scenes if scene not in VALID_SCENES]
    if invalid:
        raise ValueError(f"invalid scene profile(s): {','.join(invalid)}")
    if not scenes:
        raise ValueError("at least one scene profile is required")
    return scenes


def load_scene_profile(scene: str) -> dict[str, Any]:
    if scene not in VALID_SCENES:
        raise ValueError(f"invalid scene profile: {scene}")
    path = ROOT / "configs" / "scene_profiles" / f"{scene}.yaml"
    data = load_yaml(path)
    if data.get("scene_profile") != scene:
        raise ValueError(f"scene profile mismatch in {path}")
    return data


def expected_frames(fps: float, capture_duration_s: float) -> int:
    return int(round(float(fps) * float(capture_duration_s)))

