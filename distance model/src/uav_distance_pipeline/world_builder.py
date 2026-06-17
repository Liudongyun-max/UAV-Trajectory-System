from __future__ import annotations

from pathlib import Path

from .config import ROOT, VALID_SCENES, load_scene_profile


WORLD_SCENE_DEFAULTS = {
    "empty_clean_no_wind": {"background": "empty", "ground": "minimal", "wind": 0.0, "gust": 0.0, "direction": 0, "noise": "clean"},
    "flat_ground_light_wind": {"background": "flat_ground", "ground": "textured", "wind": 1.5, "gust": 0.5, "direction": 90, "noise": "light"},
    "forest_edge_moderate_wind": {"background": "forest_edge", "ground": "textured", "wind": 4.5, "gust": 1.5, "direction": 90, "noise": "moderate"},
}


def scene_simulation_defaults(scene: str) -> dict:
    if scene not in VALID_SCENES:
        raise ValueError(f"invalid scene: {scene}")
    return WORLD_SCENE_DEFAULTS[scene].copy()


def generate_world(scene: str, root: Path = ROOT) -> Path:
    load_scene_profile(scene)
    defaults = scene_simulation_defaults(scene)
    target = root / "sim" / "generated_worlds" / f"{scene}.world"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"""<?xml version=\"1.0\"?>
<sdf version=\"1.6\">
  <world name=\"{scene}\">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <scene>
      <ambient>0.75 0.78 0.76 1</ambient>
      <shadows>{str(scene != 'empty_clean_no_wind').lower()}</shadows>
    </scene>
    <physics name=\"default_physics\" type=\"ode\">
      <max_step_size>0.004</max_step_size>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <!-- controlled_wind mean={defaults['wind']} gust={defaults['gust']} direction_deg={defaults['direction']} -->
    <!-- sensor_noise level={defaults['noise']} render=3840x2160 output=2560x1440 fps=25 -->
  </world>
</sdf>
""", encoding="utf-8")
    return target

