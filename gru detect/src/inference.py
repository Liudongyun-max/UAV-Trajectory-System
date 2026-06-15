import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
import torch

from src.model import UAVTrajectoryNet
from src.paths import video_archive_name
from src.tracker import FeatureTracker


def _open_video_writer(path, fps, width, height):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"无法创建输出视频文件: {path}")
    return writer


def validate_video(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False, "文件不存在或大小为 0"
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, "OpenCV 无法打开输出文件"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, _ = cap.read()
        if frame_count <= 0 or not ret:
            return False, f"输出文件无有效帧: frame_count={frame_count}, first_frame={ret}"
        return True, f"校验通过: {frame_count} 帧, {os.path.getsize(path)} 字节"
    finally:
        cap.release()


class InferenceSession:
    def __init__(
        self,
        video_path,
        model_path,
        tracks_path=None,
        smooth=True,
        device="auto",
        save_video=True,
        outputs_root=None,
        show_pred=True,
        show_hist=True,
        show_gt=True,
    ):
        self.video_path = str(video_path)
        self.model_path = str(model_path)
        self.tracks_path = str(tracks_path) if tracks_path else None
        self.smooth = bool(smooth)
        self.device_arg = device
        self.save_video = bool(save_video)
        self.outputs_root = Path(outputs_root) if outputs_root else Path(__file__).resolve().parents[1] / "outputs"
        self.show_pred = bool(show_pred)
        self.show_hist = bool(show_hist)
        self.show_gt = bool(show_gt)

        self.logs = []
        self.metrics_rows = []
        self.finished = False
        self.output_path = ""
        self.temp_output_path = ""
        self.archive_name = video_archive_name(self.video_path)
        self.output_dir = self.outputs_root / self.archive_name

        self.cap = None
        self.writer = None
        self.model = None
        self.tracker = None
        self.df_measured = None
        self.df_ideal = None
        self.has_tracks = False
        self.start_idx = 0
        self.frame_idx = 0
        self.takeoff_triggered = False
        self.last_ade = 0.0

        self._open()

    def _log(self, text):
        self.logs.append(text)

    def drain_logs(self):
        logs = self.logs
        self.logs = []
        return logs

    def _open(self):
        dev = "cuda" if self.device_arg == "auto" and torch.cuda.is_available() else self.device_arg
        if dev == "auto":
            dev = "cpu"
        self.device = dev
        self._log(f"[INFO] 计算设备: {self.device}")

        self.model = UAVTrajectoryNet.load_from_checkpoint(self.model_path, device=self.device)
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        model_config = checkpoint.get("model_config", {})
        self.input_dim = model_config.get("input_dim", self.model.input_dim)
        self.future_steps = model_config.get("future_steps", self.model.future_steps)
        self.seq_len = model_config.get("seq_len", 20)
        self._log(f"[INFO] 模型加载成功: {Path(self.model_path).name}, 输入维度 {self.input_dim}, 预测步数 {self.future_steps}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开输入视频: {self.video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(self.fps) or self.fps <= 1e-3:
            self.fps = 25.0
            self._log("[WARN] 视频 FPS 异常，已按 25 FPS 处理。")
        if self.width <= 0 or self.height <= 0 or self.total_frames <= 0:
            raise RuntimeError("视频宽高或总帧数无效。")
        self._log(f"[INFO] 视频规格: {self.width}x{self.height} @ {self.fps:.3f} FPS, 共 {self.total_frames} 帧")

        self.has_tracks = self.tracks_path is not None and os.path.exists(self.tracks_path)
        if self.has_tracks:
            self.df_measured = pd.read_csv(self.tracks_path).ffill().bfill()
            required = {"x_center", "y_center"}
            missing = required - set(self.df_measured.columns)
            if missing:
                raise RuntimeError(f"测量轨迹 CSV 缺少必要列: {', '.join(sorted(missing))}")
            self.df_measured["x_center"] = float(self.width) - self.df_measured["x_center"]
            self._log(f"[INFO] 已载入测量轨迹: {self.tracks_path} (已完成水平镜像校正)")

            ideal_csv = self.tracks_path.replace("measured_tracks.csv", "ideal_tracks.csv")
            if os.path.exists(ideal_csv):
                self.df_ideal = pd.read_csv(ideal_csv)
                if "x_center" in self.df_ideal.columns:
                    self.df_ideal["x_center"] = float(self.width) - self.df_ideal["x_center"]
                if "visible" in self.df_ideal.columns:
                    visible_indices = self.df_ideal[self.df_ideal["visible"] == 1].index
                    self.start_idx = int(visible_indices[0]) if len(visible_indices) > 0 else 0
                self._log(f"[INFO] 已载入真实轨迹: {ideal_csv} (已完成水平镜像校正)")
            else:
                self._log("[WARN] 未找到 ideal_tracks.csv，仅显示预测轨迹，不计算 ADE 差值。")
        else:
            self._log("[INFO] 纯视频模式：等待目标进入画面并积累历史窗口后开始预测。")

        self.tracker = FeatureTracker(seq_len=self.seq_len, input_dim=self.input_dim, smooth=self.smooth, normalize=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.save_video:
            self.output_path = str(self.output_dir / f"{self.archive_name}_predicted.avi")
            self.temp_output_path = str(self.output_dir / f".{self.archive_name}_predicted.writing.avi")
            if os.path.exists(self.temp_output_path):
                os.remove(self.temp_output_path)
            self.writer = _open_video_writer(self.temp_output_path, self.fps, self.width, self.height)
            self._log(f"[INFO] 正在实时写入: {self.temp_output_path}")

    def step(self):
        if self.finished:
            return None
        ret, frame = self.cap.read()
        if not ret:
            self.close("执行完毕")
            return None

        rendered, ade = self._render_frame(frame)
        if self.writer is not None:
            if rendered.shape[1] != self.width or rendered.shape[0] != self.height:
                rendered = cv2.resize(rendered, (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.writer.write(rendered)

        result = {
            "frame": rendered,
            "frame_idx": self.frame_idx,
            "total_frames": self.total_frames,
            "ade": ade,
            "progress": int((self.frame_idx + 1) / max(1, self.total_frames) * 100),
        }
        self.frame_idx += 1
        return result

    def _render_frame(self, frame):
        csv_row = None
        row_valid = True
        if self.has_tracks:
            if self.frame_idx < len(self.df_measured):
                candidate = self.df_measured.iloc[self.frame_idx].to_dict()
                visible = int(candidate.get("visible", candidate.get("gt_visible", 1))) == 1
                missing = int(candidate.get("is_missing", 0)) == 1
                finite_xy = np.isfinite(float(candidate.get("x_center", np.nan))) and np.isfinite(float(candidate.get("y_center", np.nan)))
                row_valid = visible and not missing and finite_xy
                if row_valid:
                    csv_row = candidate
            else:
                row_valid = False

        if self.has_tracks and not row_valid:
            drone_pos, offset, is_valid = self.tracker.update(frame, csv_row=None)
        else:
            drone_pos, offset, is_valid = self.tracker.update(frame, csv_row)
            
        rendered = frame.copy()
        pred_coords = []
        gt_coords = []
        frame_ade = 0.0

        # 对每一个 Track 分别运行 GRU 时序推理并确认真伪
        for tid, track in self.tracker.tracks.items():
            if len(track.history_buffer) == self.seq_len:
                features_np = self.tracker.get_track_features(track)
                features_t = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, pred_offsets = self.model(features_t)
                    pred_offsets = pred_offsets.squeeze(0).cpu().numpy()
                    prob = torch.sigmoid(logits).item()
                    
                track.classification_prob = prob
                # 适当上调分类置信度判定阈值至 0.60，以排除弱噪声干扰
                track.is_uav = prob >= 0.60
                
                # 采用防抖平滑位置 smooth_px/py 作为预测坐标的起点，消除抖动
                current_x, current_y = track.smooth_px, track.smooth_py
                scale_x = self.width / 2.0
                scale_y = self.height / 2.0
                
                track_pred = []
                for off in pred_offsets:
                    track_pred.append((int(current_x + off[0] * scale_x), int(current_y + off[1] * scale_y)))
                track.pred_coords = track_pred
            else:
                track.classification_prob = None
                track.is_uav = False
                track.pred_coords = []

        # 筛选置信度最高（classification_prob最大）的唯一黄金主轨迹进行高亮展示，隐藏其它多余分支
        primary_track = None
        uav_tracks = [t for t in self.tracker.tracks.values() if t.is_uav]
        if uav_tracks:
            primary_track = max(uav_tracks, key=lambda t: t.classification_prob)

        # 寻找匹配 CSV 数据的目标，以计算 ADE 评估值
        bound_track = None
        if csv_row is not None:
            bound_tid = self.tracker._find_csv_bound_track_id(csv_row)
            if bound_tid is not None and bound_tid in self.tracker.tracks:
                bound_track = self.tracker.tracks[bound_tid]

        # 评估精度，使用绑定的 track（计算 ADE 兼容评估指标）
        if bound_track is not None and bound_track.is_uav and bound_track.pred_coords:
            pred_coords = bound_track.pred_coords
            if self.df_ideal is not None:
                gt_rows = self.df_ideal.iloc[self.frame_idx + 1 : self.frame_idx + 1 + self.future_steps]
                if {"x_center", "y_center"}.issubset(gt_rows.columns):
                    gt_coords = [(int(r["x_center"] + offset[0]), int(r["y_center"] + offset[1])) for _, r in gt_rows.iterrows()]
                    if self.show_gt:
                        for i in range(1, len(gt_coords)):
                            cv2.line(rendered, gt_coords[i - 1], gt_coords[i], (0, 255, 0), 2, cv2.LINE_AA)
                        for coord in gt_coords:
                            cv2.circle(rendered, coord, 4, (0, 255, 0), -1, cv2.LINE_AA)
                            
            if len(gt_coords) == len(pred_coords) and gt_coords:
                errors = [np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(pred_coords, gt_coords)]
                frame_ade = float(np.mean(errors))
                self.metrics_rows.append({
                    "frame": self.frame_idx,
                    "ade_px": frame_ade,
                    "pred_points": ";".join([f"{x},{y}" for x, y in pred_coords]),
                    "gt_points": ";".join([f"{x},{y}" for x, y in gt_coords]),
                })

        # 单一主轨渲染与 HUD 信息绘制 (全图只显示最高置信度的唯一 UAV)
        self._draw_history_and_hud(rendered, primary_track, frame_ade, bool(gt_coords))
        self.last_ade = frame_ade
        return rendered, frame_ade

    def _draw_history_and_hud(self, frame, primary_track, frame_ade, has_gt):
        # 1. 有且仅对置信度最高的一条黄金主轨进行画面高亮框选和连线，排除重叠分支
        if primary_track is not None and primary_track.is_uav:
            # 绘制绿色历史轨迹 (使用 static px/py)
            if self.show_hist and len(primary_track.history_buffer) > 1:
                for i in range(1, len(primary_track.history_buffer)):
                    p0 = primary_track.history_buffer[i - 1]
                    p1 = primary_track.history_buffer[i]
                    pt0 = (int(p0["px"]), int(p0["py"]))
                    pt1 = (int(p1["px"]), int(p1["py"]))
                    cv2.line(frame, pt0, pt1, (0, 69, 255), 2, cv2.LINE_AA)
                    
            # 绘制黄色 BBox 框 (使用防抖平滑后的坐标 smooth_px/py/w/h)
            cX = int(primary_track.smooth_px)
            cY = int(primary_track.smooth_py)
            w = max(30, int(primary_track.smooth_w))
            h = max(30, int(primary_track.smooth_h))
            cv2.rectangle(frame, (cX - w // 2, cY - h // 2), (cX + w // 2, cY + h // 2), (0, 255, 255), 2)
            
            # 标出唯一 UAV 的编号和分类概率
            prob = primary_track.classification_prob if primary_track.classification_prob is not None else 1.0
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"UAV #{primary_track.track_id} (conf: {prob:.2f})"
            cv2.putText(frame, text, (cX - w // 2, cY - h // 2 - 8), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            
            # 绘制未来 5 步的预测轨迹线和点
            if self.show_pred and primary_track.pred_coords:
                pred_pts = primary_track.pred_coords
                for i in range(1, len(pred_pts)):
                    cv2.line(frame, pred_pts[i - 1], pred_pts[i], (255, 255, 0), 2, cv2.LINE_AA)
                for coord in pred_pts:
                    cv2.circle(frame, coord, 4, (255, 255, 0), -1, cv2.LINE_AA)

        # 2. 统计当前帧状态（用于 HUD 打印）
        active_uavs = sum(1 for t in self.tracker.tracks.values() if t.is_uav)
        filtered_noises = sum(1 for t in self.tracker.tracks.values() if not t.is_uav and len(t.history_buffer) >= 5)

        # 3. 绘制带有磨砂黑背景的精致 HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 16), (440, 185), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Frame: {self.frame_idx}/{self.total_frames}", (34, 45), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Active UAVs: {active_uavs}", (34, 75), font, 0.52, (0, 255, 0) if active_uavs > 0 else (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Filtered Noise Tracks: {filtered_noises}", (34, 105), font, 0.52, (0, 69, 255) if filtered_noises > 0 else (180, 180, 180), 1, cv2.LINE_AA)
        
        if frame_ade > 0:
            cv2.putText(frame, f"Primary UAV ADE: {frame_ade:.2f} px", (34, 135), font, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Primary UAV ADE: N/A", (34, 135), font, 0.52, (160, 160, 160), 1, cv2.LINE_AA)
            
        cv2.putText(frame, f"GT: {'ON' if has_gt else 'OFF'}  Smooth: {'ON' if self.smooth else 'OFF'}", (34, 165), font, 0.45, (255, 180, 80), 1, cv2.LINE_AA)

    def close(self, status="执行完毕"):
        if self.finished:
            return status, self.output_path
        self.finished = True

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        final_status = status
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            ok, reason = validate_video(self.temp_output_path)
            if ok:
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                os.replace(self.temp_output_path, self.output_path)
                ok_final, final_reason = validate_video(self.output_path)
                if ok_final:
                    self._log(f"[INFO] 输出视频校验通过: {final_reason}")
                else:
                    final_status = f"错误: 输出视频归档后校验失败: {final_reason}"
                    self._log(f"[ERROR] {final_status}")
            else:
                final_status = f"错误: 输出视频校验失败: {reason}"
                self._log(f"[ERROR] {final_status}")

        if self.metrics_rows:
            try:
                metrics_path = self.output_dir / "prediction_metrics.csv"
                pd.DataFrame(self.metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
                self._log(f"[INFO] ADE 差值明细已保存: {metrics_path}")
            except Exception as exc:
                self._log(f"[WARN] ADE 差值明细保存失败: {exc}")

        return final_status, self.output_path
