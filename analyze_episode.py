import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# Parse arguments
parser = argparse.ArgumentParser(description="UAV Trajectory Data Quality Analysis Tool")
parser.add_argument("--input", "-i", required=True, help="Path to the input episode directory (e.g. export/input/episode_000001)")
parser.add_argument("--output", "-o", help="Root directory for outputs. Default: F:/UAV Trajectory System/export/outputs")
args = parser.parse_args()

INPUT_DIR = os.path.abspath(args.input)
episode_id = os.path.basename(INPUT_DIR)

if args.output:
    OUTPUT_ROOT = os.path.abspath(args.output)
else:
    # 动态定位 export 目录以自适应获取 outputs 路径
    parts = INPUT_DIR.split(os.sep)
    if "export" in parts:
        export_idx = parts.index("export")
        # 对 windows 盘符如 F: 进行重组处理
        OUTPUT_ROOT = os.path.abspath(parts[0] + os.sep + os.path.join(*parts[1:export_idx + 1], "outputs"))
    else:
        OUTPUT_ROOT = os.path.abspath(os.path.join(INPUT_DIR, "..", "..", "outputs"))

# 动态获取相对 export/input 的层级路径，实现输出与输入目录层级的镜像对齐
input_root_dir = os.path.abspath(os.path.join(os.path.dirname(OUTPUT_ROOT), "input"))
if INPUT_DIR.startswith(input_root_dir):
    rel_path = os.path.relpath(INPUT_DIR, start=input_root_dir)
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "analysis", rel_path)
else:
    parent_dir_name = os.path.basename(os.path.dirname(INPUT_DIR))
    if parent_dir_name.lower() not in ["input", "export", "analysis"]:
        OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "analysis", parent_dir_name, episode_id)
    else:
        OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "analysis", episode_id)
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"==================================================")
print(f"Analyzing UAV Episode: {episode_id}")
print(f"Input Directory:       {INPUT_DIR}")
print(f"Output Directory:      {OUTPUT_DIR}")
print(f"==================================================")

# ----------------- Helper Functions -----------------
def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

# ----------------- 1. Data Integrity Analysis -----------------
print("Running module: Data Integrity")
integrity_report = {
    "file_existence": {},
    "csv_row_counts": {},
    "anomalies": [],
    "passed": True
}

required_files = [
    "episode.yaml", "scenario.json", "camera_info.json", "frames.csv",
    "truth.csv", "ideal_tracks.csv", "measured_tracks.csv", "events.jsonl",
    "validation.json", "execution.log", "rgb.mp4", "world.world"
]

for rf in required_files:
    path = os.path.join(INPUT_DIR, rf)
    exists = os.path.exists(path)
    integrity_report["file_existence"][rf] = {
        "exists": exists,
        "size_bytes": os.path.getsize(path) if exists else 0
    }
    if not exists:
        integrity_report["anomalies"].append(f"Missing file: {rf}")
        # Not marking integrity_passed as False if non-critical files are missing
        if rf in ["frames.csv", "truth.csv", "ideal_tracks.csv", "measured_tracks.csv"]:
            integrity_report["passed"] = False

# Check if CSVs exist before reading
frames_csv_path = os.path.join(INPUT_DIR, "frames.csv")
truth_csv_path = os.path.join(INPUT_DIR, "truth.csv")
ideal_csv_path = os.path.join(INPUT_DIR, "ideal_tracks.csv")
measured_csv_path = os.path.join(INPUT_DIR, "measured_tracks.csv")

has_frames = os.path.exists(frames_csv_path)
has_truth = os.path.exists(truth_csv_path)
has_ideal = os.path.exists(ideal_csv_path)
has_measured = os.path.exists(measured_csv_path)

df_frames, df_truth, df_ideal, df_measured = None, None, None, None
csv_dfs = {}

if has_frames:
    df_frames = pd.read_csv(frames_csv_path)
    csv_dfs["frames.csv"] = df_frames
if has_truth:
    df_truth = pd.read_csv(truth_csv_path)
    csv_dfs["truth.csv"] = df_truth
if has_ideal:
    df_ideal = pd.read_csv(ideal_csv_path)
    csv_dfs["ideal_tracks.csv"] = df_ideal
if has_measured:
    df_measured = pd.read_csv(measured_csv_path)
    csv_dfs["measured_tracks.csv"] = df_measured

expected_rows = 386 # Default for first episode, but we dynamically check match if not 386
if csv_dfs:
    first_len = len(next(iter(csv_dfs.values())))
    for name, df in csv_dfs.items():
        integrity_report["csv_row_counts"][name] = len(df)
        if len(df) != first_len:
            integrity_report["anomalies"].append(f"Row count mismatch: {name} has {len(df)} rows, but others have {first_len}")
            integrity_report["passed"] = False

# Verify frame_id continuity and timestamp monotonicity
for name, df in csv_dfs.items():
    if "frame_id" not in df.columns:
        integrity_report["anomalies"].append(f"Column 'frame_id' missing in {name}")
        integrity_report["passed"] = False
        continue
    
    frame_ids = df["frame_id"].tolist()
    if len(frame_ids) > 0:
        expected_ids = list(range(frame_ids[0], frame_ids[0] + len(frame_ids)))
        if frame_ids != expected_ids:
            integrity_report["anomalies"].append(f"frame_id is not continuous or out of order in {name}")
            integrity_report["passed"] = False

    # NaN / Inf checks
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        nan_cnt = df[col].isna().sum()
        inf_cnt = np.isinf(df[col]).sum()
        if nan_cnt > 0:
            integrity_report["anomalies"].append(f"{nan_cnt} NaN values in column '{col}' of {name}")
            integrity_report["passed"] = False
        if inf_cnt > 0:
            integrity_report["anomalies"].append(f"{inf_cnt} Inf values in column '{col}' of {name}")
            integrity_report["passed"] = False

# Monotonicity of timestamps
if df_frames is not None:
    for ts_col in ["image_timestamp", "state_timestamp"]:
        if ts_col in df_frames.columns:
            diffs = df_frames[ts_col].diff().dropna()
            if (diffs < 0).any():
                integrity_report["anomalies"].append(f"Timestamp col {ts_col} in frames.csv is not monotonically increasing")
                integrity_report["passed"] = False

# Align frame_ids across CSVs
if df_frames is not None:
    ref_fids = df_frames["frame_id"].tolist()
    for name, df in csv_dfs.items():
        if name != "frames.csv":
            if df["frame_id"].tolist() != ref_fids:
                integrity_report["anomalies"].append(f"frame_id mismatch/misalignment in {name} compared to frames.csv")
                integrity_report["passed"] = False

# Video frame count consistency
rgb_path = os.path.join(INPUT_DIR, "rgb.mp4")
if os.path.exists(rgb_path) and df_frames is not None:
    cap = cv2.VideoCapture(rgb_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_frames != len(df_frames):
        integrity_report["anomalies"].append(f"Video frame count ({video_frames}) does not match frames.csv count ({len(df_frames)})")
        integrity_report["passed"] = False
else:
    video_frames = 0

with open(os.path.join(OUTPUT_DIR, "integrity_report.json"), "w", encoding="utf-8") as f:
    json.dump(integrity_report, f, indent=2)


# ----------------- 2. Time Synchronization Analysis -----------------
sync_summary = {}
if df_frames is not None:
    print("Running module: Time Synchronization")
    
    # 动态解析飞行事件时间轴
    events_path = os.path.join(INPUT_DIR, "events.jsonl")
    t_offset = None
    evt_arm = None
    evt_exec = None
    evt_post = None
    evt_land = None
    evt_finish = None
    
    sim_arm_t = 6.457
    sim_exec_t = 36.492
    sim_post_t = 66.614
    sim_land_t = 69.622
    sim_finish_t = 74.639
    
    if os.path.exists(events_path):
        events = []
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        
        # 1. 动态自适应计算 t_offset
        for e in events:
            wt = e.get("wall_time")
            sim_ts = e.get("sim_timestamp")
            if wt is not None and sim_ts is not None and sim_ts != "null":
                try:
                    sim_ts_f = float(sim_ts)
                    t_offset = wt - sim_ts_f
                    print(f"Calculated t_offset dynamically: {t_offset:.6f}s (based on event state '{e['state']}')")
                    break
                except:
                    pass
        
        # 2. 匹配获取各个关键事件的 wall_time
        for e in events:
            st = e["state"]
            wt = e.get("wall_time")
            if wt is None:
                continue
            
            if st == "MAVROS_ARM" or st == "ARM":
                evt_arm = wt
            elif st == "MAVROS_EXECUTE_MANEUVER" or st == "RECORD_ACTIVE":
                evt_exec = wt
            elif st == "MAVROS_POST_ROLL" or st == "POST_ROLL":
                evt_post = wt
            elif st == "MAVROS_LAND" or st == "LAND":
                evt_land = wt
            elif st == "FINISHED" or st == "EPISODE_FINISHED":
                evt_finish = wt
                
    if t_offset is None:
        if evt_arm is not None:
            t_offset = evt_arm - 6.457
        else:
            t_offset = 1781083342.188 # fallback
            
    if evt_arm is not None: sim_arm_t = evt_arm - t_offset
    if evt_exec is not None: sim_exec_t = evt_exec - t_offset
    if evt_post is not None: sim_post_t = evt_post - t_offset
    if evt_land is not None: sim_land_t = evt_land - t_offset
    if evt_finish is not None: sim_finish_t = evt_finish - t_offset
    
    sim_settle_t = sim_post_t
    if df_truth is not None and "world_z" in df_truth.columns:
        max_z = df_truth["world_z"].max()
        target_z = max_z * 0.975
        climb_df = df_truth[(df_truth["sim_timestamp"] >= sim_arm_t) & (df_truth["sim_timestamp"] <= sim_post_t)]
        settled_sub = climb_df[climb_df["world_z"] >= target_z]
        if len(settled_sub) > 0:
            sim_settle_t = settled_sub["sim_timestamp"].min()
            
    print(f"Dynamic time thresholds parsed:")
    print(f" - ARM:      {sim_arm_t:.3f}s")
    print(f" - EXECUTE:  {sim_exec_t:.3f}s")
    print(f" - SETTLE:   {sim_settle_t:.3f}s")
    print(f" - POST:     {sim_post_t:.3f}s")
    print(f" - LAND:     {sim_land_t:.3f}s")
    print(f" - FINISH:   {sim_finish_t:.3f}s")

    def calc_sync_metrics(df_sub):
        if len(df_sub) == 0:
            return {}
        errors = df_sub["sync_error_ms"].abs()
        valids = df_sub["sync_valid"]
        return {
            "count": int(len(df_sub)),
            "mean_ms": float(errors.mean()),
            "median_ms": float(errors.median()),
            "p90_ms": float(errors.quantile(0.90)),
            "p95_ms": float(errors.quantile(0.95)),
            "p99_ms": float(errors.quantile(0.99)),
            "max_ms": float(errors.max()),
            "invalid_ratio": float((valids == 0).mean())
        }

    if df_truth is not None and "visible" in df_truth.columns:
        df_frames["visible"] = df_truth["visible"]
    else:
        df_frames["visible"] = 0

    sync_metrics_all = calc_sync_metrics(df_frames)
    sync_metrics_no_first_15 = calc_sync_metrics(df_frames[df_frames["frame_id"] >= 15])

    t0 = df_frames["image_timestamp"].iloc[0]
    sync_metrics_no_first_1s = calc_sync_metrics(df_frames[df_frames["image_timestamp"] >= t0 + 1.0])
    sync_metrics_stable_hover = calc_sync_metrics(df_frames[df_frames["image_timestamp"] >= sim_exec_t])
    sync_metrics_visible_only = calc_sync_metrics(df_frames[df_frames["visible"] == 1])

    sync_summary = {
        "all_frames": sync_metrics_all,
        "no_first_15_frames": sync_metrics_no_first_15,
        "no_first_1s": sync_metrics_no_first_1s,
        "stable_hover_phase": sync_metrics_stable_hover,
        "visible_frames_only": sync_metrics_visible_only
    }

    # Audit 1564ms anomaly at frame_id = 0
    anomaly_frame = df_frames[df_frames["sync_error_ms"] > 500]
    if len(anomaly_frame) > 0:
        max_err_fid = anomaly_frame.iloc[0]["frame_id"]
        surrounding_frames = df_frames[(df_frames["frame_id"] >= max_err_fid - 10) & (df_frames["frame_id"] <= max_err_fid + 10)]
        surrounding_frames.to_csv(os.path.join(OUTPUT_DIR, "sync_frame_audit.csv"), index=False)

    # Plots
    plt.figure(figsize=(10, 4))
    plt.plot(df_frames["image_timestamp"], df_frames["sync_error_ms"], color='#dc3545', linewidth=1.5, label='Sync Error (ms)')
    plt.axhline(y=4.0, color='#ffc107', linestyle='--', label='95% Threshold (4ms)')
    plt.title("Time Synchronization Error Timeline", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Time (s)", fontsize=10)
    plt.ylabel("Sync Error (ms)", fontsize=10)
    plt.yscale('symlog')
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sync_error_timeline.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(df_frames["sync_error_ms"], bins=30, color='#17a2b8', edgecolor='black', alpha=0.8)
    plt.title("Time Synchronization Error Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Sync Error (ms)", fontsize=10)
    plt.ylabel("Frame Count", fontsize=10)
    plt.yscale('log')
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sync_error_histogram.png"), dpi=150)
    plt.close()


# ----------------- 3. Flight Phase & Visibility Analysis -----------------
visibility_summary = {}
if df_truth is not None and df_frames is not None:
    print("Running module: Flight Phase & Visibility")

    def get_flight_phase(t, z, vz=0.0):
        if t < sim_arm_t: return "PRE_ROLL"
        elif z < 0.1 and t < sim_settle_t: return "ARM"
        elif z >= 0.1 and t < sim_settle_t: return "TAKEOFF"
        elif t >= sim_settle_t and t < sim_post_t:
            if t < sim_settle_t + 3.0 or abs(vz) >= 0.1: return "SETTLE"
            else: return "HOVER"
        elif t >= sim_post_t and t < sim_land_t: return "POST_ROLL"
        elif t >= sim_land_t and t < sim_finish_t: return "LAND"
        else: return "UNKNOWN"

    df_frames["world_z"] = df_truth["world_z"]
    df_frames["velocity_z"] = df_truth["velocity_z"]
    df_frames["phase"] = df_frames.apply(lambda r: get_flight_phase(r["image_timestamp"], r["world_z"], r["velocity_z"]), axis=1)

    phase_stats = []
    phases_order = ["PRE_ROLL", "ARM", "TAKEOFF", "SETTLE", "HOVER", "POST_ROLL", "LAND", "UNKNOWN"]
    df_truth["phase"] = df_frames["phase"]
    df_truth["range_m"] = pd.to_numeric(df_truth["range_m"], errors='coerce')
    df_truth["projected_u"] = pd.to_numeric(df_truth["projected_u"], errors='coerce')
    df_truth["projected_v"] = pd.to_numeric(df_truth["projected_v"], errors='coerce')

    for phase in phases_order:
        sub_f = df_frames[df_frames["phase"] == phase]
        sub_t = df_truth[df_truth["phase"] == phase]
        total_cnt = len(sub_f)
        if total_cnt == 0:
            if phase == "HOVER":
                phase_stats.append({
                    "phase": phase, "total_frames": 0, "visible_frames": 0,
                    "visibility_ratio": 0.0, "sync_valid_ratio": 0.0,
                    "mean_distance_m": 0.0, "mean_pixel_u": 0.0, "mean_pixel_v": 0.0
                })
            continue
        
        vis_cnt = int(sub_t["visible"].sum())
        vis_ratio = float(vis_cnt / total_cnt)
        sync_ratio = float((sub_f["sync_valid"] == 1).mean())
        mean_dist = float(sub_t["range_m"].mean()) if total_cnt > 0 else 0.0
        mean_u = float(sub_t["projected_u"].mean()) if total_cnt > 0 else 0.0
        mean_v = float(sub_t["projected_v"].mean()) if total_cnt > 0 else 0.0
        
        phase_stats.append({
            "phase": phase, "total_frames": total_cnt, "visible_frames": vis_cnt,
            "visibility_ratio": vis_ratio, "sync_valid_ratio": sync_ratio,
            "mean_distance_m": mean_dist, "mean_pixel_u": mean_u, "mean_pixel_v": mean_v
        })

    df_phase_audit = pd.DataFrame(phase_stats)
    df_phase_audit.to_csv(os.path.join(OUTPUT_DIR, "visibility_phase_audit.csv"), index=False)

    visible_indices = df_truth[df_truth["visible"] == 1]["frame_id"].tolist()
    total_visible = len(visible_indices)

    def get_intervals(indices):
        if not indices: return [], 0
        intervals = []
        start = indices[0]
        max_len = 1
        curr_len = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                curr_len += 1
            else:
                intervals.append([int(start), int(indices[i-1])])
                max_len = max(max_len, curr_len)
                start = indices[i]
                curr_len = 1
        intervals.append([int(start), int(indices[-1])])
        max_len = max(max_len, curr_len)
        return intervals, max_len

    visible_intervals, max_visible_len = get_intervals(visible_indices)
    invisible_indices = df_truth[df_truth["visible"] == 0]["frame_id"].tolist()
    invisible_intervals, max_invisible_len = get_intervals(invisible_indices)

    vis_sync_indices = df_truth[(df_truth["visible"] == 1) & (df_truth["sync_valid"] == 1)]["frame_id"].tolist()
    _, max_vis_sync_len = get_intervals(vis_sync_indices)

    visibility_summary = {
        "total_visible_frames": total_visible,
        "visibility_ratio": float(total_visible / len(df_truth)),
        "max_continuous_visible_frames": max_visible_len,
        "max_continuous_invisible_frames": max_invisible_len,
        "max_continuous_visible_sync_valid_frames": max_vis_sync_len,
        "visible_intervals": visible_intervals,
        "invisible_intervals": invisible_intervals
    }

    # Plot Phase Visibility
    plt.figure(figsize=(10, 4))
    colors_map = {
        "PRE_ROLL": "#6c757d", "ARM": "#ffc107", "TAKEOFF": "#007bff",
        "SETTLE": "#fd7e14", "HOVER": "#28a745", "POST_ROLL": "#17a2b8", 
        "LAND": "#e83e8c", "UNKNOWN": "#343a40"
    }
    plt.step(df_frames["frame_id"], df_frames["world_z"], color='#007bff', where='mid', label='Altitude (m)')
    for phase in df_frames["phase"].unique():
        sub = df_frames[df_frames["phase"] == phase]
        plt.axvspan(sub["frame_id"].min(), sub["frame_id"].max(), alpha=0.15, color=colors_map.get(phase, '#000'), label=phase)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left")
    plt.title("Flight Phase & Altitude Profile", fontsize=12, fontweight='bold')
    plt.xlabel("Frame ID", fontsize=10)
    plt.ylabel("Altitude world_z (m)", fontsize=10)
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "phase_visibility_timeline.png"), dpi=150)
    plt.close()

    # Plot Target Visibility Timeline
    plt.figure(figsize=(10, 2))
    plt.fill_between(df_truth["frame_id"], df_truth["visible"], color='#28a745', alpha=0.6, label='Visible')
    plt.title("Target Visibility Timeline", fontsize=12, fontweight='bold')
    plt.xlabel("Frame ID", fontsize=10)
    plt.yticks([0, 1], ['Invisible', 'Visible'])
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "visibility_timeline.png"), dpi=150)
    plt.close()

    # Image position timeline
    plt.figure(figsize=(10, 4))
    plt.plot(df_truth["frame_id"], df_truth["projected_u"], color='#fd7e14', label='projected_u (X px)')
    plt.plot(df_truth["frame_id"], df_truth["projected_v"], color='#6f42c1', label='projected_v (Y px)')
    plt.axhline(y=1280, color='red', linestyle=':', label='Screen Boundary (1280x720)')
    plt.axhline(y=720, color='red', linestyle=':')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title("Target Center 2D Projection Timeline (GT)", fontsize=12, fontweight='bold')
    plt.xlabel("Frame ID", fontsize=10)
    plt.ylabel("Pixel Coordinate", fontsize=10)
    plt.legend(loc="upper left")
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "image_position_timeline.png"), dpi=150)
    plt.close()


# ----------------- 4. World Motion & Hover Quality Analysis -----------------
hover_metrics = {}
if df_truth is not None:
    print("Running module: World Motion & Hover Quality")
    df_truth["sim_timestamp"] = pd.to_numeric(df_truth["sim_timestamp"])
    df_settle = df_truth[df_truth["phase"] == "HOVER"]
    if len(df_settle) == 0:
        df_settle = df_truth[df_truth["phase"] == "SETTLE"]
    
    if len(df_settle) > 0:
        ref_x = df_settle["world_x"].mean()
        ref_y = df_settle["world_y"].mean()
        ref_z = df_settle["world_z"].mean()
        
        horizontal_drifts = np.sqrt((df_settle["world_x"] - ref_x)**2 + (df_settle["world_y"] - ref_y)**2)
        hover_metrics = {
            "settle_frames_count": len(df_settle),
            "ref_x": float(ref_x), "ref_y": float(ref_y), "ref_z": float(ref_z),
            "horizontal_rms_drift_m": float(np.sqrt((horizontal_drifts**2).mean())),
            "horizontal_p95_radius_m": float(horizontal_drifts.quantile(0.95)),
            "max_horizontal_drift_m": float(horizontal_drifts.max()),
            "height_std_m": float(df_settle["world_z"].std()),
            "max_height_deviation_m": float((df_settle["world_z"] - ref_z).abs().max()),
            "mean_horizontal_speed_mps": float(np.sqrt(df_settle["velocity_x"]**2 + df_settle["velocity_y"]**2).mean()),
            "p95_horizontal_speed_mps": float(np.sqrt(df_settle["velocity_x"]**2 + df_settle["velocity_y"]**2).quantile(0.95)),
            "net_displacement_m": float(np.sqrt((df_settle["world_x"].iloc[-1] - df_settle["world_x"].iloc[0])**2 +
                                                (df_settle["world_y"].iloc[-1] - df_settle["world_y"].iloc[0])**2 +
                                                (df_settle["world_z"].iloc[-1] - df_settle["world_z"].iloc[0])**2))
        }

    trajectory_metrics = {
        "hover_metrics": hover_metrics,
        "world_x_range": [float(df_truth["world_x"].min()), float(df_truth["world_x"].max())],
        "world_y_range": [float(df_truth["world_y"].min()), float(df_truth["world_y"].max())],
        "world_z_range": [float(df_truth["world_z"].min()), float(df_truth["world_z"].max())],
        "max_velocity_mps": float(np.sqrt(df_truth["velocity_x"]**2 + df_truth["velocity_y"]**2 + df_truth["velocity_z"]**2).max()),
        "max_acceleration_mps2": float(np.sqrt(df_truth["acceleration_x"]**2 + df_truth["acceleration_y"]**2 + df_truth["acceleration_z"]**2).max())
    }

    with open(os.path.join(OUTPUT_DIR, "trajectory_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(trajectory_metrics, f, indent=2)

    # Plot trajectories
    plt.figure(figsize=(10, 4))
    plt.plot(df_truth["sim_timestamp"], df_truth["world_x"], color='r', label='X (m)')
    plt.plot(df_truth["sim_timestamp"], df_truth["world_y"], color='g', label='Y (m)')
    plt.plot(df_truth["sim_timestamp"], df_truth["world_z"], color='b', label='Z (m)')
    plt.title("UAV World Coordinates (GT)", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Time (s)", fontsize=10)
    plt.ylabel("Position (m)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "world_position.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df_truth["sim_timestamp"], df_truth["velocity_x"], color='r', label='v_x (m/s)')
    plt.plot(df_truth["sim_timestamp"], df_truth["velocity_y"], color='g', label='v_y (m/s)')
    plt.plot(df_truth["sim_timestamp"], df_truth["velocity_z"], color='b', label='v_z (m/s)')
    plt.title("UAV World Velocities (GT)", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Time (s)", fontsize=10)
    plt.ylabel("Velocity (m/s)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "world_velocity.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df_truth["sim_timestamp"], np.degrees(df_truth["roll"]), color='purple', label='Roll (deg)')
    plt.plot(df_truth["sim_timestamp"], np.degrees(df_truth["pitch"]), color='orange', label='Pitch (deg)')
    plt.plot(df_truth["sim_timestamp"], np.degrees(df_truth["yaw"]), color='brown', label='Yaw (deg)')
    plt.title("UAV Attitude Timeline (GT)", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Time (s)", fontsize=10)
    plt.ylabel("Angle (deg)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "attitude.png"), dpi=150)
    plt.close()

    # XY scatter
    plt.figure(figsize=(6, 6))
    if len(df_settle) > 0:
        plt.scatter(df_settle["world_x"], df_settle["world_y"], color='#28a745', alpha=0.6, edgecolors='none', label='Settle Position')
        plt.scatter([hover_metrics["ref_x"]], [hover_metrics["ref_y"]], color='red', marker='X', s=100, label='Mean Center')
        plt.title("Hover Horizontal Drift Scatter (Settle Phase)", fontsize=12, fontweight='bold')
        plt.xlabel("World X (m)", fontsize=10)
        plt.ylabel("World Y (m)", fontsize=10)
        plt.legend()
        plt.grid(True, ls=":", alpha=0.5)
        plt.axis('equal')
    else:
        plt.text(0.5, 0.5, "No Settle Phase Data", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hover_xy_scatter.png"), dpi=150)
    plt.close()


# ----------------- 5. 2D Ideal Trajectory Analysis -----------------
if df_ideal is not None:
    print("Running module: 2D Ideal Trajectory")
    df_ideal_vis = df_ideal[df_ideal["visible"] == 1].copy()

    df_ideal_vis["dt"] = df_ideal_vis["timestamp"].diff().fillna(0.0)
    df_ideal_vis["pixel_vx"] = (df_ideal_vis["x_center"].diff() / df_ideal_vis["dt"]).fillna(0.0)
    df_ideal_vis["pixel_vy"] = (df_ideal_vis["y_center"].diff() / df_ideal_vis["dt"]).fillna(0.0)
    df_ideal_vis["pixel_v"] = np.sqrt(df_ideal_vis["pixel_vx"]**2 + df_ideal_vis["pixel_vy"]**2)

    plt.figure(figsize=(8, 6))
    plt.plot(df_ideal["x_center"], df_ideal["y_center"], color='#6f42c1', label='Full Episode')
    plt.scatter(df_ideal_vis["x_center"], df_ideal_vis["y_center"], color='#fd7e14', s=10, label='Visible Portion')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.title("2D Target Center Trajectory in Image Coordinates", fontsize=12, fontweight='bold')
    plt.xlabel("Image X (px)", fontsize=10)
    plt.ylabel("Image Y (px)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ideal_trajectory_xy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df_ideal["timestamp"], df_ideal["x_center"], color='orange', label='x_center (px)')
    plt.plot(df_ideal["timestamp"], df_ideal["y_center"], color='blue', label='y_center (px)')
    plt.title("2D Coordinates Timeline", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Time (s)", fontsize=10)
    plt.ylabel("Coordinates (px)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ideal_xy_timeline.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df_ideal["timestamp"], df_ideal["bbox_width"], color='m', label='width (px)')
    plt.plot(df_ideal["timestamp"], df_ideal["bbox_height"], color='c', label='height (px)')
    plt.title("Bounding Box Dimensions Timeline (GT)", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Time (s)", fontsize=10)
    plt.ylabel("Size (px)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "bbox_size_timeline.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(df_ideal_vis["timestamp"], df_ideal_vis["pixel_v"], color='brown', label='Pixel Speed (px/s)')
    plt.title("2D Pixel Center Velocity Timeline (Visible)", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Time (s)", fontsize=10)
    plt.ylabel("Pixel Speed (px/s)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pixel_velocity.png"), dpi=150)
    plt.close()


# ----------------- 6. Measurement Noise Analysis -----------------
if df_ideal is not None and df_measured is not None:
    print("Running module: Measurement Noise")
    merged = pd.merge(df_ideal, df_measured, on="frame_id", suffixes=("_ideal", "_meas"))
    merged_vis = merged[(merged["visible_ideal"] == 1) & (merged["is_missing_meas"] == 0)].copy()

    error_x = merged_vis["x_center_meas"] - merged_vis["x_center_ideal"]
    error_y = merged_vis["y_center_meas"] - merged_vis["y_center_ideal"]
    center_error = np.sqrt(error_x**2 + error_y**2)
    bbox_w_error = merged_vis["bbox_width_meas"] - merged_vis["bbox_width_ideal"]
    bbox_h_error = merged_vis["bbox_height_meas"] - merged_vis["bbox_height_ideal"]

    def get_stats(series):
        if len(series) == 0: return {}
        return {
            "mean": float(series.mean()), "median": float(series.median()), "std": float(series.std()),
            "p50": float(series.median()), "p90": float(series.quantile(0.90)), "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)), "max": float(series.max())
        }

    vis_meas = merged[merged["visible_ideal"] == 1]
    total_vis_cnt = len(vis_meas)
    missing_cnt = int(vis_meas["is_missing_meas"].sum())
    missing_ratio = float(missing_cnt / total_vis_cnt) if total_vis_cnt > 0 else 0.0

    missing_runs = []
    run_len = 0
    for idx, r in vis_meas.iterrows():
        if r["is_missing_meas"] == 1: run_len += 1
        else:
            if run_len > 0:
                missing_runs.append(run_len)
                run_len = 0
    if run_len > 0: missing_runs.append(run_len)

    id_switches = 0
    last_tid = None
    for tid in merged_vis["track_id_meas"]:
        if last_tid is not None and tid != last_tid: id_switches += 1
        last_tid = tid

    outlier_threshold = 3.0 * center_error.std() if len(center_error) > 0 else 1.0
    outliers_cnt = int((center_error > outlier_threshold).sum()) if len(center_error) > 0 else 0

    noise_metrics = {
        "error_x": get_stats(error_x), "error_y": get_stats(error_y), "center_error": get_stats(center_error),
        "bbox_width_error": get_stats(bbox_w_error), "bbox_height_error": get_stats(bbox_h_error),
        "system_bias": {
            "bias_x": float(error_x.mean()) if len(error_x) > 0 else 0.0,
            "bias_y": float(error_y.mean()) if len(error_y) > 0 else 0.0
        },
        "missing_statistics": {
            "visible_frames_count": total_vis_cnt, "missing_frames_count": missing_cnt, "missing_ratio": missing_ratio,
            "consecutive_missing_runs_distribution": [int(x) for x in missing_runs],
            "max_consecutive_missing": int(max(missing_runs)) if missing_runs else 0
        },
        "id_switches": id_switches, "outliers_count": outliers_cnt,
        "confidence_statistics": get_stats(merged_vis["detector_confidence_meas"])
    }

    with open(os.path.join(OUTPUT_DIR, "noise_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(noise_metrics, f, indent=2)

    # Plot Noise
    plt.figure(figsize=(10, 5))
    plt.plot(merged["frame_id"], merged["x_center_ideal"], color='blue', linestyle='-', label='Ideal Center X')
    plt.plot(merged["frame_id"], merged["y_center_ideal"], color='red', linestyle='-', label='Ideal Center Y')
    df_meas_vis = merged[merged["is_missing_meas"] == 0]
    plt.scatter(df_meas_vis["frame_id"], df_meas_vis["x_center_meas"], color='cyan', marker='x', s=15, alpha=0.8, label='Measured X')
    plt.scatter(df_meas_vis["frame_id"], df_meas_vis["y_center_meas"], color='magenta', marker='x', s=15, alpha=0.8, label='Measured Y')
    plt.title("Ideal vs Measured Target Centers Overlay", fontsize=12, fontweight='bold')
    plt.xlabel("Frame ID", fontsize=10)
    plt.ylabel("Coordinates (px)", fontsize=10)
    plt.legend(loc="upper left")
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "ideal_measured_overlay.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(merged_vis["frame_id"], center_error, color='red', label='Center Translation Error (px)')
    plt.title("Measurement Center Translation Error Timeline", fontsize=12, fontweight='bold')
    plt.xlabel("Frame ID", fontsize=10)
    plt.ylabel("Error (px)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "center_error_timeline.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(center_error, bins=25, color='orange', edgecolor='black', alpha=0.8)
    plt.title("Center Translation Error Histogram", fontsize=12, fontweight='bold')
    plt.xlabel("Error (px)", fontsize=10)
    plt.ylabel("Frame Count", fontsize=10)
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "center_error_histogram.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(error_x, error_y, color='purple', alpha=0.6, edgecolors='none', label='Noise Error (Meas - GT)')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Measurement Noise Error Scatter (X vs Y)", fontsize=12, fontweight='bold')
    plt.xlabel("Error X (px)", fontsize=10)
    plt.ylabel("Error Y (px)", fontsize=10)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "error_xy_scatter.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(merged_vis["frame_id"], merged_vis["detector_confidence_meas"], color='green', label='Detector Confidence')
    plt.title("Detector Confidence Timeline", fontsize=12, fontweight='bold')
    plt.xlabel("Frame ID", fontsize=10)
    plt.ylabel("Confidence", fontsize=10)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confidence_timeline.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 2))
    plt.fill_between(merged["frame_id"], merged["is_missing_meas"], color='gray', alpha=0.5, label='Measured Missing')
    plt.title("Measured Missing Timeline", fontsize=12, fontweight='bold')
    plt.xlabel("Frame ID", fontsize=10)
    plt.yticks([0, 1], ['Detected', 'Missing'])
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "missing_timeline.png"), dpi=150)
    plt.close()


# ----------------- 7. 20+5 Window Audit -----------------
window_audit_summary = {}
if df_frames is not None and df_truth is not None and df_ideal is not None and df_measured is not None:
    print("Running module: 20+5 Window Audit")
    df_merged_all = pd.merge(df_frames, df_truth, on="frame_id", suffixes=("", "_truth"))
    df_merged_all = pd.merge(df_merged_all, df_ideal, on="frame_id", suffixes=("", "_ideal"))
    df_merged_all = pd.merge(df_merged_all, df_measured, on="frame_id", suffixes=("", "_meas"))

    total_windows = len(df_merged_all) - 25 + 1
    window_records = []

    for start_fid in range(total_windows):
        end_fid = start_fid + 24
        sub_df = df_merged_all[(df_merged_all["frame_id"] >= start_fid) & (df_merged_all["frame_id"] <= end_fid)].copy()
        
        is_continuous = len(sub_df) == 25 and (sub_df["frame_id"].tolist() == list(range(start_fid, end_fid + 1)))
        track_ids = sub_df["track_id_ideal"].unique()
        is_same_track = len(track_ids) == 1 and track_ids[0] == 1
        phases = sub_df["phase"].unique()
        is_same_phase = len(phases) == 1
        
        future_df = sub_df.iloc[20:25]
        future_gt_exists = (future_df["is_missing"] == 0).all()
        future_visible = (future_df["visible_ideal"] == 1).all()
        future_sync_valid = (future_df["sync_valid"] == 1).all()
        
        hist_df = sub_df.iloc[0:20]
        hist_sync_ratio = (hist_df["sync_valid"] == 1).mean()
        hist_sync_ok = hist_sync_ratio >= 0.95
        hist_valid_meas_cnt = ((hist_df["visible_ideal"] == 1) & (hist_df["is_missing_meas"] == 0)).sum()
        hist_meas_ok = hist_valid_meas_cnt >= 18
        
        consec_missing = 0
        max_consec_missing = 0
        for idx, r in hist_df.iterrows():
            if r["visible_ideal"] == 1 and r["is_missing_meas"] == 1:
                consec_missing += 1
                max_consec_missing = max(max_consec_missing, consec_missing)
            else:
                consec_missing = 0
        max_consec_missing = max(max_consec_missing, consec_missing)
        hist_consec_missing_ok = max_consec_missing <= 2
        
        has_anomaly_frame = 0 in sub_df["frame_id"].values
        contains_pre_post = ("PRE_ROLL" in phases) or ("ARM" in phases) or ("LAND" in phases) or ("UNKNOWN" in phases)
        
        status = "VALID"
        reasons = []
        
        if not is_continuous:
            status = "INVALID_TIMESTAMP"
            reasons.append("Timestamp discontinuity")
        elif not is_same_track:
            status = "INVALID_TRACK_SWITCH"
            reasons.append("Track ID switch")
        elif contains_pre_post:
            status = "INVALID_PHASE_CROSSING"
            reasons.append("Contains pre-takeoff/post-landing phases")
        elif not is_same_phase:
            status = "INVALID_PHASE_CROSSING"
            reasons.append("Crosses flight phases")
        elif has_anomaly_frame:
            status = "INVALID_SYNC"
            reasons.append("Contains 1564ms sync anomaly frame")
        elif not future_gt_exists:
            status = "INVALID_NO_FUTURE_GT"
            reasons.append("Future 5 frames lack ideal GT")
        elif not future_visible or not future_sync_valid:
            status = "INVALID_INVISIBLE" if not future_visible else "INVALID_SYNC"
            if not future_visible: reasons.append("Future 5 frames not visible")
            if not future_sync_valid: reasons.append("Future 5 frames have invalid sync")
        elif not hist_sync_ok:
            status = "INVALID_SYNC"
            reasons.append(f"History sync_valid ratio ({hist_sync_ratio:.2f}) < 95%")
        elif not hist_meas_ok:
            ideal_vis_cnt = (hist_df["visible_ideal"] == 1).sum()
            if ideal_vis_cnt < 18:
                status = "INVALID_INVISIBLE"
                reasons.append(f"Drone visible in only {ideal_vis_cnt} frames (requires >=18)")
            else:
                status = "INVALID_LONG_MISSING"
                reasons.append(f"Valid measurements in hist ({hist_valid_meas_cnt}) < 18")
        elif not hist_consec_missing_ok:
            status = "INVALID_LONG_MISSING"
            reasons.append(f"Max consecutive missing ({max_consec_missing}) > 2")
            
        window_records.append({
            "window_id": start_fid, "start_frame_id": start_fid, "end_frame_id": end_fid,
            "start_timestamp": sub_df["image_timestamp"].iloc[0], "end_timestamp": sub_df["image_timestamp"].iloc[-1],
            "status": status, "reasons": "; ".join(reasons) if reasons else "None", "phases_present": ", ".join(phases),
            "hist_sync_valid_ratio": hist_sync_ratio, "hist_valid_meas_count": int(hist_valid_meas_cnt),
            "hist_max_consec_missing": int(max_consec_missing), "future_visible": int(future_visible),
            "future_sync_valid": int(future_sync_valid)
        })

    df_window_audit = pd.DataFrame(window_records)
    df_window_audit.to_csv(os.path.join(OUTPUT_DIR, "window_audit.csv"), index=False)

    df_valid = df_window_audit[df_window_audit["status"] == "VALID"]
    df_valid.to_csv(os.path.join(OUTPUT_DIR, "valid_windows.csv"), index=False)

    df_invalid = df_window_audit[df_window_audit["status"] != "VALID"]
    df_invalid.to_csv(os.path.join(OUTPUT_DIR, "invalid_windows.csv"), index=False)

    def calc_valid_by_stride(stride):
        valid_cnt = 0
        total_cnt = 0
        for w_id in range(0, total_windows, stride):
            total_cnt += 1
            if window_records[w_id]["status"] == "VALID": valid_cnt += 1
        return total_cnt, valid_cnt

    tot_1, val_1 = calc_valid_by_stride(1)
    tot_3, val_3 = calc_valid_by_stride(3)
    tot_5, val_5 = calc_valid_by_stride(5)
    status_counts = df_window_audit["status"].value_counts().to_dict()

    window_audit_summary = {
        "candidate_windows_count": len(df_window_audit),
        "valid_windows_count": len(df_valid),
        "invalid_windows_count": len(df_invalid),
        "valid_ratio": float(len(df_valid) / len(df_window_audit)) if len(df_window_audit) > 0 else 0.0,
        "categories_breakdown": {str(k): int(v) for k, v in status_counts.items()},
        "stride_comparison": {
            "stride_1": {"total": tot_1, "valid": val_1, "ratio": float(val_1/tot_1) if tot_1 > 0 else 0.0},
            "stride_3": {"total": tot_3, "valid": val_3, "ratio": float(val_3/tot_3) if tot_3 > 0 else 0.0},
            "stride_5": {"total": tot_5, "valid": val_5, "ratio": float(val_5/tot_5) if tot_5 > 0 else 0.0}
        }
    }


# ----------------- 8. Write Summary JSON -----------------
    episode_yaml_path = os.path.join(INPUT_DIR, "episode.yaml")
    meta_mission = "hover"
    meta_distance = 300.0
    meta_fallback = False
    meta_capture_mode = "gazebo_live"
    if os.path.exists(episode_yaml_path):
        try:
            import yaml
            with open(episode_yaml_path, "r", encoding="utf-8") as f:
                ey = yaml.safe_load(f)
                meta_mission = ey.get("mission", "hover")
                meta_distance = float(ey.get("distance_m", 300.0))
                meta_fallback = bool(ey.get("fallback_used", False))
                meta_capture_mode = ey.get("capture_mode", "gazebo_live")
        except Exception as ex:
            print(f"Error reading episode.yaml: {ex}")

    summary_json = {
        "episode_id": episode_id,
        "distance_m": meta_distance,
        "mission": meta_mission,
        "capture_mode": meta_capture_mode,
        "fallback_used": meta_fallback,
        "total_frames": len(df_frames) if df_frames is not None else 0,
        "visible_frames_count": visibility_summary.get("total_visible_frames", 0),
        "integrity_passed": bool(integrity_report["passed"]),
        "sync_analysis": sync_summary,
        "visibility_analysis": visibility_summary,
        "hover_quality": hover_metrics,
        "window_audit_summary": window_audit_summary
    }

with open(os.path.join(OUTPUT_DIR, "analysis_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2)

print("\n==================================================")
print(f"Analysis for {episode_id} completed successfully!")
print(f"==================================================")
