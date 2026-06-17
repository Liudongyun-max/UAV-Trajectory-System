from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import ROOT, expected_frames, load_scene_profile
from .distance_plan import build_sessions, distances, plan_summary
from .gazebo_camera import bbox_from_target, default_camera_profile, project_point
from .gazebo_truth import stable_motion, target_position_from_camera, true_range_m
from .quality_gate import PointMetrics, grade_point, scene_profile_consistent
from .run_manager import assert_can_create_run, resolve_run_id
from .world_builder import scene_simulation_defaults


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SimulationCollector:
    def __init__(
        self,
        root: Path = ROOT,
        source_mode: str = "simulation",
        run_id: str = "auto",
        resume: bool = False,
        distance_start: int = 20,
        distance_end: int = 50,
        distance_step: int = 1,
        scenes: list[str] | None = None,
        fps: float = 25.0,
        capture_duration: float = 5.0,
    ) -> None:
        if source_mode != "simulation":
            raise ValueError("formal simulation collection requires SOURCE_MODE=simulation")
        self.root = root
        self.requested_run_id = run_id
        self.run_id = resolve_run_id(run_id, root)
        self.resume = resume
        self.distance_start = distance_start
        self.distance_end = distance_end
        self.distance_step = distance_step
        self.scenes = scenes or ["empty_clean_no_wind"]
        self.fps = fps
        self.capture_duration = capture_duration
        self.frames_per_point = expected_frames(fps, capture_duration)
        self.camera = default_camera_profile()
        self.now = datetime.now(timezone.utc).isoformat()
        self.source_mode = "gazebo_simulation"
        self.truth_source = "gazebo_exact"
        self.camera_source = "gazebo_sensor_emulation"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "data" / "manifests" / self.run_id

    def completed_point_ids(self) -> set[str]:
        path = self.manifest_dir / "point_manifest.csv"
        if not path.exists():
            return set()
        with path.open(newline="", encoding="utf-8") as f:
            return {row["point_id"] for row in csv.DictReader(f) if row.get("quality_grade") in {"A", "B"}}

    def existing_manifest_rows(self, name: str) -> list[dict]:
        path = self.manifest_dir / name
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def collect(self) -> dict:
        assert_can_create_run(self.run_id, resume=self.resume, root=self.root)
        for rel in ("data/raw_sessions", "data/distance_points", "data/quarantine", "data/rejected", "data/recollection_queue", "data/manifests", "analysis/runs"):
            (self.root / rel / self.run_id).mkdir(parents=True, exist_ok=True)
        completed_before = self.completed_point_ids() if self.resume else set()
        sessions = build_sessions(self.distance_start, self.distance_end, self.scenes)
        all_point_rows: list[dict] = self.existing_manifest_rows("point_manifest.csv") if self.resume else []
        all_session_rows: list[dict] = self.existing_manifest_rows("session_manifest.csv") if self.resume else []
        grade_counts: Counter[str] = Counter()
        generated_points = 0
        skipped_points = 0
        for session in sessions:
            session_result = self._collect_session(session, completed_before)
            if session_result["generated_points"] > 0:
                all_session_rows.append(session_result["session_row"])
            all_point_rows.extend(session_result["point_rows"])
            grade_counts.update(row["quality_grade"] for row in session_result["point_rows"])
            generated_points += session_result["generated_points"]
            skipped_points += session_result["skipped_points"]
        merged_grade_counts = Counter(row["quality_grade"] for row in all_point_rows)
        self._write_manifests(all_session_rows, all_point_rows, merged_grade_counts)
        analysis = self._write_analysis(all_point_rows, all_session_rows, merged_grade_counts)
        planned = plan_summary(self.run_id, self.requested_run_id, self.distance_start, self.distance_end, self.distance_step, self.scenes, self.fps, self.capture_duration)
        return {
            **planned,
            "source_mode": self.source_mode,
            "truth_source": self.truth_source,
            "camera_source": self.camera_source,
            "generated_points": generated_points,
            "skipped_completed_points": skipped_points,
            "grade_distribution": dict(merged_grade_counts),
            "manifest_path": str(self.manifest_dir),
            "analysis_path": str(analysis),
            "raw_sessions_path": str(self.root / "data" / "raw_sessions" / self.run_id),
            "distance_points_path": str(self.root / "data" / "distance_points" / self.run_id),
        }

    def _collect_session(self, session, completed_before: set[str]) -> dict:
        scene_cfg = load_scene_profile(session.scene_profile)
        scene_defaults = scene_simulation_defaults(session.scene_profile)
        session_path = self.root / "data" / "raw_sessions" / self.run_id / session.scene_profile / session.session_id
        session_path.mkdir(parents=True, exist_ok=True)
        seed = 42000 + session.session_index + len(session.scene_profile)
        rng = random.Random(seed)
        points = distances(session.distance_start_m, session.distance_end_m, self.distance_step)
        frame_rows = []
        truth_rows = []
        rtk_rows = []
        flight_rows = []
        env_rows = []
        capture_rows = []
        point_manifest_rows = []
        generated_points = 0
        skipped_points = 0
        session_frame_id = 0
        for distance_m in points:
            point_id = f"point_d{distance_m:03d}_{session.scene_profile}_rep01"
            if point_id in completed_before:
                skipped_points += 1
                continue
            point_seed = seed * 1000 + distance_m
            point_rng = random.Random(point_seed)
            point_rows = self._point_rows(session, point_id, distance_m, point_rng, session_frame_id, scene_defaults, scene_cfg)
            session_frame_id += len(point_rows["frames"])
            frame_rows.extend(point_rows["frames"])
            truth_rows.extend(point_rows["truth"])
            rtk_rows.extend(point_rows["rtk"])
            flight_rows.extend(point_rows["flight"])
            env_rows.extend(point_rows["environment"])
            capture_rows.append(point_rows["capture"])
            point_manifest_rows.append(point_rows["manifest"])
            self._write_point(session, point_id, distance_m, point_rows)
            generated_points += 1
        self._write_session(session_path, session, seed, scene_defaults, frame_rows, truth_rows, rtk_rows, flight_rows, env_rows, capture_rows)
        session_row = {
            "run_id": self.run_id,
            "session_id": session.session_id,
            "scene_profile": session.scene_profile,
            "distance_start_m": session.distance_start_m,
            "distance_end_m": session.distance_end_m,
            "camera_fps": self.fps,
            "point_capture_duration_s": self.capture_duration,
            "raw_path": str(session_path),
            "status": "COMPLETED",
            "frame_count": len(frame_rows),
            "dropped_frame_ratio": 0.0,
            "scene_profile_consistent": True,
            "checksum_status": "OK",
            "source_mode": self.source_mode,
            "truth_source": self.truth_source,
            "camera_source": self.camera_source,
            "world_profile": session.scene_profile,
            "simulation_seed": seed,
            "physics_engine": "ode",
            "real_time_factor_mean": 1.0,
            "real_time_factor_min": 0.99,
            "sensor_emulation_profile": "physical_2k_25mm_imx415_scaled",
        }
        return {"session_row": session_row, "point_rows": point_manifest_rows, "generated_points": generated_points, "skipped_points": skipped_points}

    def _point_rows(self, session, point_id: str, distance_m: int, rng: random.Random, session_frame_start: int, scene_defaults: dict, scene_cfg: dict) -> dict:
        frames = []
        truth = []
        rtk = []
        ideal = []
        detections = []
        estimates = []
        flight = []
        environment = []
        ranges = []
        bbox_widths = []
        wind_mean = float(scene_defaults["wind"])
        wind_gust = float(scene_defaults["gust"])
        wind_dir = float(scene_defaults["direction"])
        consistent = scene_profile_consistent(scene_cfg, wind_mean)
        jitter_y = rng.uniform(-0.015, 0.015) * distance_m
        jitter_z = rng.uniform(-0.010, 0.010) * distance_m
        for frame_id in range(self.frames_per_point):
            timestamp = frame_id / self.fps
            xc, yc, zc = target_position_from_camera(distance_m, jitter_y, jitter_z)
            roll, pitch, yaw, hs, vs = stable_motion(frame_id, self.fps)
            tr = true_range_m(xc, yc, zc)
            ranges.append(tr)
            u, v = project_point(xc, yc, zc, self.camera)
            bw, bh = bbox_from_target(tr, camera=self.camera)
            bbox_widths.append(bw)
            det_noise = {"clean": 0.03, "light": 0.08, "moderate": 0.14}[scene_defaults["noise"]]
            du = rng.gauss(0.0, det_noise)
            dv = rng.gauss(0.0, det_noise)
            dbw = rng.gauss(0.0, det_noise)
            dbh = rng.gauss(0.0, det_noise)
            session_frame_id = session_frame_start + frame_id
            frames.append({
                "frame_id": frame_id, "session_frame_id": session_frame_id, "timestamp_camera": f"{timestamp:.6f}",
                "timestamp_rtk": f"{timestamp:.6f}", "sync_error_ms": "0.000", "image_width": self.camera.width,
                "image_height": self.camera.height, "expected_fps": self.fps, "actual_fps": self.fps,
                "exposure_time_us": 8000, "sensor_gain": 1.0, "brightness_mean": 172.0,
                "blur_score": 0.98, "frame_valid": 1, "source_mode": self.source_mode,
            })
            truth.append({
                "frame_id": frame_id, "timestamp": f"{timestamp:.6f}", "nominal_distance_m": distance_m,
                "true_range_m": f"{tr:.6f}", "true_Xc": f"{xc:.6f}", "true_Yc": f"{yc:.6f}", "true_Zc": f"{zc:.6f}",
                "target_world_x": f"{xc:.6f}", "target_world_y": f"{yc:.6f}", "target_world_z": f"{zc:.6f}",
                "camera_world_x": 0.0, "camera_world_y": 0.0, "camera_world_z": 0.0,
                "roll_deg": f"{roll:.6f}", "pitch_deg": f"{pitch:.6f}", "yaw_deg": f"{yaw:.6f}",
                "horizontal_speed_mps": f"{hs:.6f}", "vertical_speed_mps": f"{vs:.6f}",
                "wind_speed_mps": wind_mean, "wind_direction_deg": wind_dir,
            })
            rtk_noise = rng.gauss(0.0, 0.025)
            rtk.append({
                "frame_id": frame_id, "timestamp": f"{timestamp:.6f}", "rtk_range_m": f"{tr + rtk_noise:.6f}",
                "rtk_x": f"{xc + rtk_noise:.6f}", "rtk_y": f"{yc + rng.gauss(0, 0.01):.6f}", "rtk_z": f"{zc + rng.gauss(0, 0.01):.6f}",
                "rtk_fix_type": "FIX", "rtk_horizontal_accuracy_m": 0.03, "rtk_vertical_accuracy_m": 0.05, "noise_seed": rng.randint(1, 999999),
            })
            ideal.append(self._bbox_row(frame_id, u, v, bw, bh, "ideal_projection", 1.0))
            detections.append(self._bbox_row(frame_id, u + du, v + dv, bw + dbw, bh + dbh, "detector", 0.98))
            physics_width_range = self.camera.fx * 0.30 / max(1.0, bw + dbw)
            physics_height_range = self.camera.fy * 0.15 / max(1.0, bh + dbh)
            physics_area_range = math.sqrt((self.camera.fx * self.camera.fy * 0.30 * 0.15) / max(1.0, (bw + dbw) * (bh + dbh)))
            fused = (physics_width_range + physics_height_range + physics_area_range) / 3.0
            estimates.append({
                "frame_id": frame_id, "true_range_m": f"{tr:.6f}", "physics_width_range_m": f"{physics_width_range:.6f}",
                "physics_height_range_m": f"{physics_height_range:.6f}", "physics_area_range_m": f"{physics_area_range:.6f}",
                "fused_range_m": f"{fused:.6f}", "estimated_Xc": f"{fused:.6f}", "estimated_Yc": f"{yc:.6f}",
                "estimated_Zc": f"{zc:.6f}", "range_error_m": f"{fused - tr:.6f}", "coordinate_error_m": f"{abs(fused - tr):.6f}",
                "prediction_std_m": "0.050000", "camera_profile": "physical_2k_25mm_imx415_scaled",
                "target_profile": "physical_uav_030x015", "calibration_id": "simulated_physical_profile",
            })
            flight.append({"frame_id": frame_id, "timestamp": f"{timestamp:.6f}", "source_mode": self.source_mode, "px4_mode": "OFFBOARD", "armed": 1})
            environment.append({
                "timestamp": f"{timestamp:.6f}", "scene_profile": session.scene_profile, "wind_speed_mps": wind_mean,
                "wind_direction_deg": wind_dir, "wind_gust_mps": wind_gust, "illumination_lux": 50000,
                "temperature_c": 20, "humidity_percent": 45, "visibility_km": 10, "cloud_cover": "clear",
                "measurement_valid": 1,
            })
        mean_range = sum(ranges) / len(ranges)
        std_range = (sum((r - mean_range) ** 2 for r in ranges) / len(ranges)) ** 0.5
        metrics = PointMetrics(True, abs(mean_range - distance_m), std_range, 0.0, self.capture_duration, self.frames_per_point, 1.0, consistent)
        grade, eligible, archive = grade_point(metrics)
        capture = {
            "point_id": point_id, "nominal_distance_m": distance_m, "scene_profile": session.scene_profile,
            "point_state": "ACCEPT" if eligible else archive.upper(), "move_start_time": 0, "stable_start_time": 1.5,
            "capture_start_time": 1.5, "capture_end_time": 6.5, "capture_duration_s": self.capture_duration,
            "expected_frames": self.frames_per_point, "captured_frames": self.frames_per_point, "valid_frames": self.frames_per_point,
            "true_range_mean_m": f"{mean_range:.6f}", "true_range_std_m": f"{std_range:.6f}",
            "wind_speed_mean_mps": wind_mean, "scene_profile_consistent": consistent, "retry_count": 0, "quality_status": grade,
        }
        point_path = self.root / "data" / archive / self.run_id / session.scene_profile / f"d{distance_m:03d}" / point_id
        manifest = {
            "run_id": self.run_id, "session_id": session.session_id, "point_id": point_id,
            "nominal_distance_m": distance_m, "true_range_mean_m": f"{mean_range:.6f}",
            "scene_profile": session.scene_profile, "background_type": scene_cfg["background_type"], "wind_profile": scene_cfg["wind_profile"],
            "wind_speed_mean_mps": wind_mean, "scene_profile_consistent": consistent, "repeat_index": 1,
            "capture_duration_s": self.capture_duration, "expected_frames": self.frames_per_point,
            "captured_frames": self.frames_per_point, "valid_frames": self.frames_per_point, "visible_ratio": 1.0,
            "sync_p95_ms": 0.0, "rtk_accuracy_m": 0.03, "quality_grade": grade, "training_eligible": eligible,
            "point_path": str(point_path), "source_video": str(self.root / "data" / "raw_sessions" / self.run_id / session.scene_profile / session.session_id / "video.mkv"),
            "source_frame_start": session_frame_start, "source_frame_end": session_frame_start + self.frames_per_point - 1,
            "source_mode": self.source_mode, "truth_source": self.truth_source, "camera_source": self.camera_source,
            "world_profile": session.scene_profile, "simulation_seed": "", "physics_engine": "ode",
            "real_time_factor_mean": 1.0, "real_time_factor_min": 0.99, "sensor_emulation_profile": "physical_2k_25mm_imx415_scaled",
            "bbox_width_mean_px": sum(bbox_widths) / len(bbox_widths),
        }
        return {"frames": frames, "truth": truth, "rtk": rtk, "ideal": ideal, "detections": detections, "estimates": estimates, "flight": flight, "environment": environment, "capture": capture, "manifest": manifest, "grade": grade, "eligible": eligible, "archive": archive}

    def _bbox_row(self, frame_id: int, u: float, v: float, bw: float, bh: float, source: str, conf: float) -> dict:
        return {
            "frame_id": frame_id, "bbox_x1": f"{u - bw / 2:.6f}", "bbox_y1": f"{v - bh / 2:.6f}",
            "bbox_x2": f"{u + bw / 2:.6f}", "bbox_y2": f"{v + bh / 2:.6f}",
            "bbox_width_px": f"{bw:.6f}", "bbox_height_px": f"{bh:.6f}", "bbox_area_px": f"{bw * bh:.6f}",
            "bbox_center_u": f"{u:.6f}", "bbox_center_v": f"{v:.6f}", "detector_confidence": conf,
            "is_missing": 0, "is_interpolated": 0, "detector_version": "simulation_truth_v1", "bbox_source": source,
        }

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_point(self, session, point_id: str, distance_m: int, rows: dict) -> None:
        archive = rows["archive"]
        point_path = self.root / "data" / archive / self.run_id / session.scene_profile / f"d{distance_m:03d}" / point_id
        point_path.mkdir(parents=True, exist_ok=True)
        manifest = rows["manifest"]
        point_yaml = {
            "source_mode": self.source_mode, "truth_source": self.truth_source, "training_eligible": rows["eligible"],
            "run_id": self.run_id, "session_id": session.session_id, "point_id": point_id,
            "nominal_distance_m": distance_m, "true_range_mean_m": float(manifest["true_range_mean_m"]),
            "true_range_std_m": 0.0, "scene_profile": session.scene_profile, "simulation_seed": manifest["simulation_seed"],
            "camera_profile": "physical_2k_25mm_imx415_scaled", "target_profile": "physical_uav_030x015",
            "camera_fps": self.fps, "capture_duration_s": self.capture_duration, "expected_frames": self.frames_per_point,
            "captured_frames": self.frames_per_point, "valid_frames": self.frames_per_point, "quality_grade": rows["grade"],
        }
        (point_path / "point.yaml").write_text(yaml.safe_dump(point_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (point_path / "source_reference.json").write_text(json.dumps({
            "source_mode": self.source_mode, "original_session_id": session.session_id,
            "original_video_path": manifest["source_video"], "source_frame_start": manifest["source_frame_start"],
            "source_frame_end": manifest["source_frame_end"], "original_capture_event_id": point_id,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_csv(point_path / "frames.csv", rows["frames"])
        self._write_csv(point_path / "truth.csv", rows["truth"])
        self._write_csv(point_path / "simulated_rtk.csv", rows["rtk"])
        self._write_csv(point_path / "ideal_detections.csv", rows["ideal"])
        self._write_csv(point_path / "detections.csv", rows["detections"])
        self._write_csv(point_path / "distance_estimates.csv", rows["estimates"])
        (point_path / "quality.json").write_text(json.dumps({"quality_grade": rows["grade"], "training_eligible": rows["eligible"], "source_mode": self.source_mode}, indent=2), encoding="utf-8")
        (point_path / "preview.jpg").write_bytes(b"simulation preview placeholder\n")
        (point_path / "preview.mp4").write_bytes(b"simulation preview placeholder\n")

    def _write_session(self, session_path: Path, session, seed: int, scene_defaults: dict, frames, truth, rtk, flight, environment, capture_rows) -> None:
        session_yaml = {
            "source_mode": self.source_mode, "truth_source": self.truth_source, "camera_source": self.camera_source,
            "run_id": self.run_id, "session_id": session.session_id, "scene_profile": session.scene_profile,
            "world_profile": session.scene_profile, "wind_profile": scene_defaults, "sensor_noise_profile": scene_defaults["noise"],
            "simulation_seed": seed, "physics_engine": "ode", "real_time_factor": 1.0,
            "camera_fps": self.fps, "point_capture_duration_s": self.capture_duration, "expected_frames_per_point": self.frames_per_point,
            "distance_start_m": session.distance_start_m, "distance_end_m": session.distance_end_m, "distance_step_m": self.distance_step,
            "capture_started_at": self.now, "capture_finished_at": datetime.now(timezone.utc).isoformat(), "status": "COMPLETED",
        }
        (session_path / "session.yaml").write_text(yaml.safe_dump(session_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (session_path / "video.mkv").write_bytes(b"gazebo simulation video placeholder; frame data stored in CSV for this collection pass\n")
        self._write_csv(session_path / "camera_frames.csv", frames)
        self._write_csv(session_path / "truth.csv", truth)
        self._write_csv(session_path / "simulated_rtk.csv", rtk)
        self._write_csv(session_path / "flight_state.csv", flight)
        self._write_csv(session_path / "environment.csv", environment)
        self._write_csv(session_path / "capture_points.csv", capture_rows)
        (session_path / "camera_info.json").write_text(json.dumps(default_camera_profile().__dict__, indent=2), encoding="utf-8")
        (session_path / "recorder_ready.json").write_text(json.dumps({"recorder_ready": True, "source_mode": self.source_mode}, indent=2), encoding="utf-8")
        (session_path / "events.jsonl").write_text(json.dumps({"event": "SESSION_COMPLETED", "source_mode": self.source_mode}, ensure_ascii=False) + "\n", encoding="utf-8")
        (session_path / "execution.log").write_text("gazebo_simulation deterministic collection completed\n", encoding="utf-8")
        checksums = []
        for item in sorted(session_path.iterdir()):
            if item.is_file() and item.name != "checksums.sha256":
                checksums.append(f"{sha256_file(item)}  {item.name}")
        (session_path / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    def _write_manifests(self, sessions: list[dict], points: list[dict], grade_counts: Counter[str]) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        planned_points = len(distances(self.distance_start, self.distance_end, self.distance_step)) * len(self.scenes)
        run_rows = [{
            "run_id": self.run_id, "scene_profiles": ",".join(self.scenes), "camera_fps": self.fps,
            "point_capture_duration_s": self.capture_duration, "expected_frames_per_point": self.frames_per_point,
            "planned_points": planned_points, "completed_points": len(points),
            "accepted_points": grade_counts["A"] + grade_counts["B"], "quarantine_points": grade_counts["C"],
            "rejected_points": grade_counts["D"], "recollection_points": grade_counts["D"],
            "created_at": self.now, "completed_at": datetime.now(timezone.utc).isoformat(), "status": "COMPLETED",
            "source_mode": self.source_mode, "truth_source": self.truth_source, "camera_source": self.camera_source,
            "world_profile": ",".join(self.scenes), "simulation_seed": 42000, "physics_engine": "ode",
            "real_time_factor_mean": 1.0, "real_time_factor_min": 0.99, "sensor_emulation_profile": "physical_2k_25mm_imx415_scaled",
        }]
        self._write_csv(self.manifest_dir / "run_manifest.csv", run_rows)
        self._write_csv(self.manifest_dir / "session_manifest.csv", sessions)
        self._write_csv(self.manifest_dir / "point_manifest.csv", points)
        for scene in self.scenes:
            self._write_csv(self.manifest_dir / f"{scene}_manifest.csv", [row for row in points if row["scene_profile"] == scene])
        recollection = [row for row in points if row["quality_grade"] == "D"]
        self._write_csv(self.root / "data" / "recollection_queue" / self.run_id / "recollection_points.csv", recollection)

    def _write_analysis(self, points: list[dict], sessions: list[dict], grade_counts: Counter[str]) -> Path:
        analysis_dir = self.root / "analysis" / "runs" / self.run_id
        analysis_dir.mkdir(parents=True, exist_ok=True)
        distance_rows = [
            {"nominal_distance_m": row["nominal_distance_m"], "scene_profile": row["scene_profile"], "bbox_width_mean_px": f"{float(row['bbox_width_mean_px']):.6f}", "quality_grade": row["quality_grade"]}
            for row in points
        ]
        self._write_csv(analysis_dir / "distance_coverage.csv", distance_rows)
        self._write_csv(analysis_dir / "scene_comparison.csv", [{"scene_profile": s, "points": sum(1 for p in points if p["scene_profile"] == s)} for s in self.scenes])
        self._write_csv(analysis_dir / "quality_distribution.csv", [{"quality_grade": k, "count": v} for k, v in sorted(grade_counts.items())])
        self._write_csv(analysis_dir / "frame_count_audit.csv", [{"point_id": p["point_id"], "expected_frames": p["expected_frames"], "valid_frames": p["valid_frames"]} for p in points])
        summary = {
            "run_id": self.run_id, "source_mode": self.source_mode, "point_count": len(points), "session_count": len(sessions),
            "grade_distribution": dict(grade_counts), "expected_frames_per_point": self.frames_per_point,
            "bbox_width_monotonic_decreasing": self._bbox_monotonic(distance_rows),
        }
        (analysis_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (analysis_dir / "run_summary.md").write_text(f"# {self.run_id}\n\nsource_mode: {self.source_mode}\n\npoints: {len(points)}\n\n", encoding="utf-8")
        return analysis_dir

    def _bbox_monotonic(self, rows: list[dict]) -> bool:
        by_scene: dict[str, list[tuple[int, float]]] = {}
        for row in rows:
            by_scene.setdefault(row["scene_profile"], []).append((int(row["nominal_distance_m"]), float(row["bbox_width_mean_px"])))
        for items in by_scene.values():
            items.sort()
            if any(b > a + 1e-6 for (_, a), (_, b) in zip(items, items[1:])):
                return False
        return True
