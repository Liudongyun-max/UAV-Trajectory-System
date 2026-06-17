#!/usr/bin/env python3
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Segoe UI Emoji', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import cv2
import torch
import yaml
import hashlib
import time
sys.path.insert(0, str(Path(__file__).parent / "train"))
from training.models.gru_distance_stabilizer import GRUDistanceStabilizer
from segmentation_mode.tracker import FeatureTracker
from yolo_mode.detector import YOLOv5Detector

class DistanceMeasurement:
    def __init__(self, raw_unclipped_range_m: float, robust_fused_range_m: float, measurement_valid: bool, measurement_quality: float, invalid_reason: str = "", candidate_ranges: list = None, candidate_qualities: list = None):
        self.raw_unclipped_range_m = raw_unclipped_range_m
        self.robust_fused_range_m = robust_fused_range_m
        self.measurement_valid = measurement_valid
        self.measurement_quality = measurement_quality
        self.invalid_reason = invalid_reason
        self.candidate_ranges = candidate_ranges or []
        self.candidate_qualities = candidate_qualities or []


class ConsoleRedirector:
    """重定向 stdout 到 Tkinter Text 控件"""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass


class UAVDistanceStabilizerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("UAV 测距模型稳定与时序视频分析系统 UI")
        self.root.geometry("850x720")
        
        # 现代暗黑背景风格设置
        self.bg_color = "#1e1e1e"
        self.fg_color = "#f0f0f0"
        self.accent_color = "#0078d4"
        self.btn_bg = "#2d2d2d"
        self.entry_bg = "#333333"
        self.label_bg = "#1e1e1e"
        
        self.root.configure(bg=self.bg_color)
        
        # 线程锁与合成状态
        self.is_running = False
        self.output_video_path = ""
        
        # 构建 UI 组件
        self.create_widgets()
        
        # 初始化加载默认数据包中的 Point IDs
        self.root.after(500, self.auto_load_points_on_start)

    def create_widgets(self):
        # 样式定义
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("TLabelframe", background=self.bg_color, foreground=self.fg_color, bordercolor="#555")
        style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.fg_color)
        style.configure("TButton", background=self.btn_bg, foreground=self.fg_color, bordercolor="#555")
        style.map("TButton", background=[("active", self.accent_color)])
        style.configure("TProgressbar", thickness=15, troughcolor="#333", background=self.accent_color)
        
        # 1. 标题
        title_label = tk.Label(
            self.root, 
            text="UAV 测距残差修正与 GRU 时序稳定视频可视化系统", 
            font=("Microsoft YaHei", 14, "bold"), 
            bg=self.bg_color, 
            fg="#00a3ff"
        )
        title_label.pack(pady=10)
        
        # 2. 文件导入区域
        file_frame = ttk.LabelFrame(self.root, text=" 核心数据与模型导入 ")
        file_frame.pack(fill="x", padx=15, pady=5)
        
        # 特征预测包 (CSV)
        tk.Label(file_frame, text="特征预测 CSV 包:", bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.csv_path_var = tk.StringVar(value="F:/UAV Trajectory System/distance model/exports/gru_input/run017/frame_predictions.csv")
        self.csv_entry = tk.Entry(file_frame, textvariable=self.csv_path_var, width=65, bg=self.entry_bg, fg=self.fg_color, insertbackground="white")
        self.csv_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览...", command=self.browse_csv).grid(row=0, column=2, padx=5, pady=5)
        
        # 视频文件 (可选，空时自动对齐)
        tk.Label(file_frame, text="指定视频文件 (可选):", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.video_path_var = tk.StringVar(value="")
        self.video_entry = tk.Entry(file_frame, textvariable=self.video_path_var, width=65, bg=self.entry_bg, fg=self.fg_color, insertbackground="white")
        self.video_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览...", command=self.browse_video).grid(row=1, column=2, padx=5, pady=5)
        
        # GRU 稳定模型权重 (.pth)
        tk.Label(file_frame, text="GRU 稳定性模型 (.pth):", bg=self.bg_color, fg=self.fg_color).grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.model_path_var = tk.StringVar(value="F:/UAV Trajectory System/distance model/models/best.pth")
        self.model_entry = tk.Entry(file_frame, textvariable=self.model_path_var, width=65, bg=self.entry_bg, fg=self.fg_color, insertbackground="white")
        self.model_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览...", command=self.browse_model).grid(row=2, column=2, padx=5, pady=5)
        
        # 配置文件 (.yaml)
        tk.Label(file_frame, text="训练配置文件 (.yaml):", bg=self.bg_color, fg=self.fg_color).grid(row=3, column=0, sticky="w", padx=10, pady=5)
        default_yaml = "F:/UAV Trajectory System/train/configs/distance_gru_static_25f.yaml"
        if not os.path.exists(default_yaml):
            alt_yaml = "F:/UAV Trajectory System/distance model/train/configs/distance_gru_static_25f.yaml"
            if os.path.exists(alt_yaml):
                default_yaml = alt_yaml
        self.config_path_var = tk.StringVar(value=default_yaml)
        self.config_entry = tk.Entry(file_frame, textvariable=self.config_path_var, width=65, bg=self.entry_bg, fg=self.fg_color, insertbackground="white")
        self.config_entry.grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="浏览...", command=self.browse_config).grid(row=3, column=2, padx=5, pady=5)
        
        # 纯视频实时预测复选框
        self.pure_video_mode_var = tk.BooleanVar(value=False)
        self.pure_video_check = tk.Checkbutton(
            file_frame, text="纯视频实时预测模式 (不加载 CSV，支持 YOLO/传统视觉算法实时预测)", 
            variable=self.pure_video_mode_var, bg=self.bg_color, fg=self.fg_color, 
            selectcolor=self.entry_bg, activebackground=self.bg_color, activeforeground=self.fg_color,
            command=self.on_pure_video_mode_toggle
        )
        self.pure_video_check.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # 纯视频检测追踪器选择
        tk.Label(file_frame, text="检测追踪器模式:", bg=self.bg_color, fg=self.fg_color).grid(row=5, column=0, sticky="w", padx=10, pady=5)
        self.detector_mode_var = tk.StringVar(value="YOLOv5 实时目标框定")
        self.detector_mode_combo = ttk.Combobox(
            file_frame, textvariable=self.detector_mode_var,
            values=["YOLOv5 实时目标框定", "图像分割+位移追踪"],
            state="disabled", width=40
        )
        self.detector_mode_combo.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.detector_mode_combo.bind("<<ComboboxSelected>>", self.on_detector_mode_changed)
        
        # 3. 参数配置区域
        param_frame = ttk.LabelFrame(self.root, text=" 运行参数配置 ")
        param_frame.pack(fill="x", padx=15, pady=5)
        
        # 点位 ID 选择
        tk.Label(param_frame, text="选择点位 (Point ID):", bg=self.bg_color, fg=self.fg_color).grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.point_combo = ttk.Combobox(param_frame, width=40, state="readonly")
        self.point_combo.grid(row=0, column=1, columnspan=2, sticky="w", padx=5, pady=5)
        ttk.Button(param_frame, text="重新扫描点位", command=self.scan_points).grid(row=0, column=3, padx=10, pady=5)
        
        # 输出 FPS，DPI 以及输出保存目录
        tk.Label(param_frame, text="视频帧率 (FPS):", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.fps_var = tk.DoubleVar(value=25.0)
        tk.Entry(param_frame, textvariable=self.fps_var, width=15, bg=self.entry_bg, fg=self.fg_color).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        tk.Label(param_frame, text="图表 DPI:", bg=self.bg_color, fg=self.fg_color).grid(row=1, column=2, sticky="w", padx=10, pady=5)
        self.dpi_var = tk.IntVar(value=100)
        tk.Entry(param_frame, textvariable=self.dpi_var, width=15, bg=self.entry_bg, fg=self.fg_color).grid(row=1, column=3, sticky="w", padx=5, pady=5)
        
        tk.Label(param_frame, text="输出保存目录:", bg=self.bg_color, fg=self.fg_color).grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.output_dir_var = tk.StringVar(value="F:/UAV Trajectory System/distance model/outputs (自动与输入物理路径严格对齐)")
        self.output_entry = tk.Entry(param_frame, textvariable=self.output_dir_var, width=65, bg=self.entry_bg, fg=self.fg_color)
        self.output_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5)
        ttk.Button(param_frame, text="浏览...", command=self.browse_output_dir).grid(row=2, column=3, padx=5, pady=5)
        
        # 4. 控制和进度区域
        control_frame = tk.Frame(self.root, bg=self.bg_color)
        control_frame.pack(fill="x", padx=15, pady=10)
        
        self.run_btn = ttk.Button(control_frame, text="开始视频渲染与误差分析", width=25, command=self.start_processing)
        self.run_btn.pack(side="left", padx=10)
        
        self.play_btn = ttk.Button(control_frame, text="播放生成的对比视频", width=25, state="disabled", command=self.play_output_video)
        self.play_btn.pack(side="left", padx=10)
        
        self.progress_bar = ttk.Progressbar(control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(side="right", padx=10, pady=5)
        
        # 5. 控制台日志文本显示区域
        log_frame = ttk.LabelFrame(self.root, text=" 实时运行日志 ")
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.log_text = tk.Text(log_frame, bg="#121212", fg="#00ff00", insertbackground="white", state="disabled", font=("Consolas", 10))
        self.log_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # stdout 重定向
        sys.stdout = ConsoleRedirector(self.log_text)
        sys.stderr = ConsoleRedirector(self.log_text)

    # 浏览事件绑定
    def browse_csv(self):
        f = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if f:
            self.csv_path_var.set(f)
            self.scan_points()

    def browse_video(self):
        f = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi")])
        if f:
            self.video_path_var.set(f)
            self.on_video_path_changed()

    def find_session_video(self, root_dir: Path, session_id: str) -> Path | None:
        if not root_dir.exists():
            return None
        for r, dirs, files in os.walk(str(root_dir)):
            if Path(r).name == session_id and "video.mkv" in files:
                return Path(r) / "video.mkv"
        return None

    def on_video_path_changed(self):
        video_path = self.video_path_var.get().strip()
        if not video_path or not os.path.exists(video_path):
            return
            
        path_obj = Path(video_path)
        distance_model_dir = Path("F:/UAV Trajectory System/distance model")
        
        # 1. 自动从视频路径中检索并适配特征预测 CSV 的 run_id
        run_id = ""
        for part in path_obj.parts:
            if part.startswith("run"):
                run_id = part
                break
                
        if run_id:
            # 拼接对应的全局特征 CSV 路径
            auto_csv_path = distance_model_dir / "exports" / "gru_input" / run_id / "frame_predictions.csv"
            if auto_csv_path.exists():
                old_csv = self.csv_path_var.get()
                if old_csv != str(auto_csv_path):
                    self.csv_path_var.set(str(auto_csv_path))
                    print(f"\n[联动适配] 检测到视频属于实验 {run_id}，已将特征预测 CSV 包自动定位更新为:\n{auto_csv_path}")
            else:
                # 备用：如果未找到 frame_predictions，检测是否是在 point 目录下的 detections.csv
                point_dir_csv = path_obj.parent / "detections.csv"
                if point_dir_csv.exists():
                    self.csv_path_var.set(str(point_dir_csv))
                    print(f"\n[联动适配] 已将特征预测 CSV 包自动定位更新为单点检测框文件:\n{point_dir_csv}")

        # 2. 尝试从路径中识别 session 标识以锁定多点候选
        session_id = ""
        if path_obj.parent.name.startswith("session_"):
            session_id = path_obj.parent.name
        else:
            for part in path_obj.parts:
                if part.startswith("session_"):
                    session_id = part
                    break
                    
        if not session_id:
            # 如果是点包目录下的 preview 视频，直接锁定唯一的 Point ID
            if path_obj.parent.name.startswith("point_"):
                point_id = path_obj.parent.name
                print(f"[联动适配] 检测到您导入了单点视频，已自动锁定 Point ID: {point_id}")
                self._update_combo_values([point_id])
                return
            return
            
        print(f"[联动适配] 关联的后台 Session ID 为: {session_id}")
        
        csv_path = self.csv_path_var.get()
        if not os.path.exists(csv_path):
            return
            
        # 扫描该 session 关联 of Point IDs 并自动限制下拉列表的候选范围
        print(f"正在当前特征 CSV 数据包中检索属于该视频 Session 的所有点位...")
        def _match_points():
            try:
                df = pd.read_csv(csv_path, usecols=["point_id", "session_id"])
                sub_df = df[df["session_id"] == session_id]
                points = sorted(sub_df["point_id"].dropna().unique())
                
                if points:
                    points_with_all = ["[All Points (Session Whole Video)]"] + points
                    self.root.after(0, lambda: self._update_combo_values(points_with_all))
                    print(f"[联动适配] 匹配成功! 已自动将点位选择框限制为该视频 Session 所关联的 {len(points)} 个点位，并开启了 Session 全视频对齐渲染模式。")
                else:
                    print(f"[联动适配] 提示: 未能在当前特征 CSV 中检索到关联 Session '{session_id}' 的数据点。")
            except Exception as e:
                print(f"[联动适配] 匹配点位失败: {str(e)}")
                
        threading.Thread(target=_match_points, daemon=True).start()

    def browse_model(self):
        f = filedialog.askopenfilename(filetypes=[("PyTorch Weight Files", "*.pth *.pt")])
        if f:
            self.model_path_var.set(f)

    def browse_config(self):
        f = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
        if f:
            self.config_path_var.set(f)

    def browse_output_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir_var.set(d)

    # 扫描 CSV 并提取 point_id 列表
    def scan_points(self):
        if hasattr(self, "pure_video_mode_var") and self.pure_video_mode_var.get():
            return
        csv_path = self.csv_path_var.get()
        if not os.path.exists(csv_path):
            print(f"提示: 未能找到 CSV 特征文件，路径为 {csv_path}。请手动指定正确路径。")
            return
        
        print(f"正在扫描 {os.path.basename(csv_path)} 中的点位数据...")
        
        def _scan():
            try:
                # 只读取一列加快解析速度
                df = pd.read_csv(csv_path, usecols=["point_id"])
                points = sorted(df["point_id"].dropna().unique())
                
                # 在主线程更新 GUI 控件
                self.root.after(0, lambda: self._update_combo_values(points))
                print(f"扫描完毕! 成功提取 {len(points)} 个有效点位 (Point ID)。")
            except Exception as e:
                print(f"解析 CSV 发生错误: {str(e)}")
        
        threading.Thread(target=_scan, daemon=True).start()

    def log_event(self, event_type: str, details: dict):
        reports_dir = getattr(self, "reports_dir", Path("F:/UAV Trajectory System/distance model/outputs/reports/runtime_distance_stability"))
        reports_dir.mkdir(parents=True, exist_ok=True)
        log_path = reports_dir / "runtime_events.jsonl"
        
        event_data = {
            "event_type": event_type,
            "timestamp": time.time(),
            "details": details
        }
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(event_data, ensure_ascii=False) + "\n")

    def reset_gru_state(self, reason: str):
        self.features_queue = []
        self.log_event("GRU_STATE_RESET", {
            "reason": reason,
            "timestamp": time.time()
        })
        print(f"   --> [GRU State Reset] 原因: {reason}")

    def show_schema_mismatch_error(self):
        def _show():
            messagebox.showerror("MODEL_SCHEMA_MISMATCH", "模型特征契约哈希校验失败！\n请检查 feature_schema.json 是否被非法修改。")
        self.root.after(0, _show)

    def on_pure_video_mode_toggle(self):
        is_pure = self.pure_video_mode_var.get()
        if is_pure:
            self.csv_entry.configure(state="disabled")
            self.point_combo.configure(state="disabled")
            self.detector_mode_combo.configure(state="readonly")
            self.point_combo.set(f"[Pure Video mode - {self.detector_mode_var.get()}]")
            print(f"--> [GUI] 切换到纯视频实时预测模式，检测器: {self.detector_mode_var.get()}，CSV 输入已禁用。")
        else:
            self.csv_entry.configure(state="normal")
            self.point_combo.configure(state="readonly")
            self.detector_mode_combo.configure(state="disabled")
            self.scan_points()
            print("--> [GUI] 切回普通数据校正模式。")

    def on_detector_mode_changed(self, event=None):
        if self.pure_video_mode_var.get():
            self.point_combo.set(f"[Pure Video mode - {self.detector_mode_var.get()}]")
            print(f"--> [GUI] 纯视频预测模式下切换检测器为: {self.detector_mode_var.get()}")

    def _update_combo_values(self, points: list[str]):
        self.point_combo["values"] = points
        if points:
            self.point_combo.set(points[0])

    def auto_load_points_on_start(self):
        self.scan_points()

    # 视频播放
    def play_output_video(self):
        if self.output_video_path and os.path.exists(self.output_video_path):
            print(f"正在使用系统默认播放器打开视频: {self.output_video_path}")
            os.startfile(self.output_video_path)
        else:
            messagebox.showerror("错误", "输出的视频文件不存在，请先运行生成。")

    # 开始视频渲染主方法 (在新线程中运行)
    def start_processing(self):
        if self.is_running:
            return
            
        is_pure = hasattr(self, "pure_video_mode_var") and self.pure_video_mode_var.get()
        
        if not is_pure:
            point_id = self.point_combo.get()
            if not point_id:
                messagebox.showwarning("警告", "请先选择一个有效的点位 (Point ID)！")
                return
            csv_path = self.csv_path_var.get()
            if not os.path.exists(csv_path):
                messagebox.showerror("错误", f"特征 CSV 包不存在，请检查路径:\n{csv_path}")
                return
        else:
            point_id = "[Pure Video mode - YOLOv5]"
            csv_path = ""
            user_video = self.video_path_var.get().strip()
            if not user_video or not os.path.exists(user_video):
                messagebox.showerror("错误", "在纯视频实时预测模式下，必须指定有效的视频文件！")
                return
            
        model_path = self.model_path_var.get()
        config_path = self.config_path_var.get()
        
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"GRU 模型文件不存在，请检查路径:\n{model_path}")
            return
        if not os.path.exists(config_path):
            messagebox.showerror("错误", f"配置文件不存在，请检查路径:\n{config_path}")
            return
            
        self.is_running = True
        self.run_btn.configure(state="disabled")
        self.play_btn.configure(state="disabled")
        self.progress_bar["value"] = 0
        
        # 启动后台线程执行合成
        t = threading.Thread(
            target=self.process_worker, 
            args=(csv_path, model_path, config_path, point_id), 
            daemon=True
        )
        t.start()

    def find_point_dir(self, source_root: Path, point_id: str) -> Path | None:
        """递归查找 point_id 对应的目录"""
        for p in source_root.rglob(point_id):
            if p.is_dir() and (p / "point.yaml").exists():
                return p
        return None

    def convert_video_path(self, original_path: str, local_raw_sessions_root: Path) -> Path | None:
        """将原始 Linux 视频路径转换为本地 Windows 路径"""
        parts = Path(original_path).parts
        try:
            idx = parts.index("raw_sessions")
            rel_path = Path(*parts[idx + 1:])
            local_path = local_raw_sessions_root / rel_path
            if local_path.exists():
                return local_path
        except ValueError:
            pass
        
        p_path = Path(original_path)
        session_id = p_path.parent.name
        for mkv in local_raw_sessions_root.rglob("video.mkv"):
            if mkv.parent.name == session_id:
                return mkv
        return None

    # 后台线程执行函数
    def process_worker(self, csv_path: str, model_path: str, config_path: str, point_id: str):
        # 内部辅助函数
        def load_mlp_model(m_path: Path) -> dict:
            data = np.load(m_path, allow_pickle=True)
            x_std = data["x_std"]
            # 保护性防溢出：若标准差极小（说明在仿真训练集上该特征为常量或接近常量），则将其设为 1.0，防止除零或除微小值引起爆栈
            x_std = np.where(x_std < 1e-2, 1.0, x_std)
            params = {
                "x_mean": data["x_mean"],
                "x_std": x_std,
                "y_mean": data["y_mean"],
                "y_std": data["y_std"],
                "w1": data["w1"],
                "b1": data["b1"],
                "w2": data["w2"],
                "b2": data["b2"],
                "feature_columns": list(data["feature_columns"]),
            }
            if "w3" in data:
                params["w3"] = data["w3"]
                params["b3"] = data["b3"]
                params["num_layers"] = 2
            else:
                params["num_layers"] = 1
            return params

        def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(x, 0.0)

        def predict_mlp(params: dict, x: np.ndarray) -> np.ndarray:
            xs = (x - params["x_mean"]) / params["x_std"]
            if params["num_layers"] == 2:
                h1 = relu(xs @ params["w1"] + params["b1"])
                h2 = relu(h1 @ params["w2"] + params["b2"])
                pred_scaled = h2 @ params["w3"] + params["b3"]
            else:
                h = relu(xs @ params["w1"] + params["b1"])
                pred_scaled = h @ params["w2"] + params["b2"]
            return (pred_scaled * params["y_std"] + params["y_mean"]).reshape(-1)

        try:
            print("\n" + "="*50)
            print("  开始执行视频合成与测距误差分析任务...")
            print("="*50)
            
            # 读取运行时防抖配置文件 (Configs)
            run_cfg_path = Path("F:/UAV Trajectory System/distance model/configs/runtime_distance_stabilization.yaml")
            if run_cfg_path.exists():
                with open(run_cfg_path, "r", encoding="utf-8") as rf:
                    run_cfg = yaml.safe_load(rf)
            else:
                run_cfg = {}

            # Schema 契约校验 (P1 阶段)
            models_dir = Path("F:/UAV Trajectory System/distance model/models")
            schema_path = models_dir / "feature_schema.json"
            if not schema_path.exists():
                print("错误: 缺少特征定义文件 feature_schema.json")
                self.log_event("SCHEMA_MISMATCH", {"reason": "missing feature_schema.json"})
                self.show_schema_mismatch_error()
                self.processing_finished(False)
                return
                
            try:
                with open(schema_path, "r", encoding="utf-8") as sf:
                    schema_data = json.load(sf)
                
                # 计算运行时哈希
                normalized_schema_str = json.dumps(schema_data, sort_keys=True)
                runtime_feature_schema_hash = hashlib.sha256(normalized_schema_str.encode("utf-8")).hexdigest()
                
                # 读取合同哈希
                contract_json_path = Path("F:/UAV Trajectory System/distance model/reports/runtime_distance_stability/runtime_feature_contract.json")
                if contract_json_path.exists():
                    with open(contract_json_path, "r", encoding="utf-8") as cf:
                        contract_data = json.load(cf)
                    expected_hash = contract_data.get("model_feature_schema_hash", "")
                    if runtime_feature_schema_hash != expected_hash:
                        raise ValueError(f"Runtime feature schema hash ({runtime_feature_schema_hash}) does not match expected model hash ({expected_hash})!")
            except Exception as ex:
                print(f"错误: Feature Schema 契约校验失败: {ex}")
                self.log_event("SCHEMA_MISMATCH", {"reason": str(ex)})
                self.show_schema_mismatch_error()
                self.processing_finished(False)
                return

            is_pure_video = hasattr(self, "pure_video_mode_var") and self.pure_video_mode_var.get()
            detector_mode = self.detector_mode_var.get() if hasattr(self, "detector_mode_var") else "YOLOv5 实时目标框定"
            
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                
            train_out_dir = Path("F:/UAV Trajectory System/train/outputs") / cfg["experiment"]["name"]
            if not train_out_dir.exists():
                alt_train_out_dir = Path("F:/UAV Trajectory System/distance model/train/outputs") / cfg["experiment"]["name"]
                if alt_train_out_dir.exists():
                    train_out_dir = alt_train_out_dir
            distance_model_dir = Path("F:/UAV Trajectory System/distance model")
            local_raw_sessions_root = distance_model_dir / "input" / "raw_sessions"
            
            is_session_mode = (point_id == "[All Points (Session Whole Video)]")
            session_id = ""
            df_pt = pd.DataFrame()
            
            if is_pure_video:
                # 1. 纯视频模式：直接加载视频
                print("1. [纯视频模式] 绕过特征 CSV 数据包加载。")
                user_video = self.video_path_var.get().strip()
                if not user_video or not os.path.exists(user_video):
                    print("错误: 纯视频模式下必须指定有效的视频文件。")
                    self.processing_finished(False)
                    return
                video_file = Path(user_video)
                
                # 获取视频元数据
                cap_temp = cv2.VideoCapture(str(video_file))
                if not cap_temp.isOpened():
                    print("错误: 无法打开指定的视频文件。")
                    self.processing_finished(False)
                    return
                total_video_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap_temp.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                cap_temp.release()
                
                start_frame = 0
                end_frame = total_video_frames - 1
                total_render_frames = total_video_frames
                
                print(f"2. [纯视频模式] 已定位视频源: {video_file}，视频渲染物理帧范围: {start_frame} 至 {end_frame} (总渲染步长: {total_render_frames} 帧，FPS={fps:.2f})")
                
                # 根据不同的检测追踪器模式，选择性地加载模型
                if detector_mode == "YOLOv5 实时目标框定":
                    # 通过模块化包装器加载 YOLOv5 模型
                    print("   --> 正在加载 YOLOv5 实时目标定位器 (包装器)...")
                    yolo_weight_path = "F:/UAV Trajectory System/yolov5_model/best include small.pt"
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    detector = YOLOv5Detector(model_path=yolo_weight_path, conf_thres=0.3, device=device)
                else:
                    print("   --> [图像分割模式] 采用传统视觉分割与位移时序追踪方案 (无须加载神经网络模型参数，秒级启动)。")
                    detector = None
                
                # 加载 NumPy MLP 残差预测权重
                print("   --> 正在加载 NumPy Residual MLP 模型...")
                mlp_model_path = distance_model_dir / "models" / "distance_residual_mlp_numpy.npz"
                mlp_params = load_mlp_model(mlp_model_path)
                mlp_feature_cols = mlp_params["feature_columns"]
                
            else:
                # 普通模式：读取数据包并过滤指定的 Point ID 或整个 Session ID 的并集
                print("1. 正在加载并过滤时序特征数据...")
                df_all = pd.read_csv(csv_path)
                
                # 自动寻找 point_manifest.csv 路径以解析物理视频中的真实帧对齐
                manifest_csv_path = Path(csv_path).parent / "point_manifest.csv"
                point_start_map = {}
                point_end_map = {}
                if manifest_csv_path.exists():
                    try:
                        df_manifest = pd.read_csv(manifest_csv_path, usecols=["point_id", "source_frame_start", "source_frame_end"])
                        point_start_map = dict(zip(df_manifest["point_id"], df_manifest["source_frame_start"]))
                        point_end_map = dict(zip(df_manifest["point_id"], df_manifest["source_frame_end"]))
                    except Exception as e:
                        print(f"   警告: 读取 point_manifest.csv 失败: {e}")
                
                if is_session_mode:
                    user_video = self.video_path_var.get().strip()
                    if user_video:
                        path_obj = Path(user_video)
                        if path_obj.parent.name.startswith("session_"):
                            session_id = path_obj.parent.name
                        else:
                            for part in path_obj.parts:
                                if part.startswith("session_"):
                                    session_id = part
                                    break
                    if not session_id:
                        combo_vals = list(self.point_combo["values"])
                        if len(combo_vals) > 1:
                            sample_pt = combo_vals[1]
                            sample_row = df_all[df_all["point_id"] == sample_pt]
                            if not sample_row.empty:
                                session_id = sample_row.iloc[0]["session_id"]
                    
                    if not session_id:
                        print("错误: 无法确定当前视频对应的 Session ID。")
                        self.processing_finished(False)
                        return
                        
                    print(f"   [Session 全程模式] 关联的 Session ID 为: {session_id}")
                    df_pt = df_all[df_all["session_id"] == session_id].copy()
                else:
                    df_pt = df_all[df_all["point_id"] == point_id].copy()
                    
                if df_pt.empty:
                    print(f"错误: 没有找到对应的数据，退出。")
                    self.processing_finished(False)
                    return
                    
                # 2. 对齐原始视频路径与全局物理帧号
                user_video = self.video_path_var.get().strip()
                video_file = None
                start_frame = 0
                
                if is_session_mode:
                    if user_video and os.path.exists(user_video):
                        print(f"2. 用户手动指定视频输入: {user_video}")
                        video_file = Path(user_video)
                    else:
                        print("2. 正在自动定位 Session 原始飞行动画视频...")
                        video_file = self.find_session_video(local_raw_sessions_root, session_id)
                else:
                    if user_video and os.path.exists(user_video):
                        print(f"2. 用户手动指定视频输入: {user_video}")
                        video_file = Path(user_video)
                        # 如果用户手动指定了视频，我们依然需要获取 start_frame
                        point_dir = self.find_point_dir(distance_model_dir / "input" / "distance_points", point_id)
                        if point_dir:
                            with (point_dir / "source_reference.json").open("r", encoding="utf-8") as f:
                                ref_info = json.load(f)
                            start_frame = int(ref_info["original_start_frame_id"])
                    else:
                        print("2. 正在自动解析点位配置并对齐原始 Gazebo 仿真视频...")
                        point_dir = self.find_point_dir(distance_model_dir / "input" / "distance_points", point_id)
                        if point_dir:
                            with (point_dir / "source_reference.json").open("r", encoding="utf-8") as f:
                                ref_info = json.load(f)
                            start_frame = int(ref_info["original_start_frame_id"])
                            orig_video_path = ref_info["original_video_path"]
                            video_file = self.convert_video_path(orig_video_path, local_raw_sessions_root)
                
                # 使用 point_start_map 对局部 frame_id 进行全局物理帧映射
                # 如果 map 里找不到 (比如是未注册的本地单点 CSV)，则 fallback 使用 start_frame 偏移量
                def get_global_frame(row):
                    pt = row["point_id"]
                    offset = point_start_map.get(pt, start_frame)
                    return int(row["frame_id"] + offset)
                    
                df_pt["global_frame_id"] = df_pt.apply(get_global_frame, axis=1)
                
                # 去除可能重复 of global_frame_id 行
                df_pt = df_pt.drop_duplicates(subset=["global_frame_id"]).sort_values("global_frame_id").reset_index(drop=True)
                n_frames = len(df_pt)
                
                start_frame = int(df_pt["global_frame_id"].min())
                end_frame = int(df_pt["global_frame_id"].max())
                total_render_frames = end_frame - start_frame + 1
                
                print(f"   已定位视频源: {video_file}，视频渲染物理帧范围: {start_frame} 至 {end_frame} (总渲染步长: {total_render_frames} 帧)")
                
                if not video_file or not video_file.exists():
                    print("   警告: 原始飞行动画视频 mkv 未找到。系统将降级为全黑画布展示。")
                    video_file = None
                
            # 3. 加载 GRU 权重并在当前特征流上做前向预测
            print("3. 加载 GRU 稳定模型，进行实时测距残差时序滤波推理...")
            models_dir = distance_model_dir / "models"
            with (models_dir / "feature_schema.json").open("r", encoding="utf-8") as f:
                schema = json.load(f)
            feature_cols = schema["feature_columns"]
            input_dim = schema["input_dim"]
            
            with (models_dir / "normalization.json").open("r", encoding="utf-8") as f:
                norm_info = json.load(f)
            means = np.array([norm_info["mean"][c] for c in feature_cols], dtype=np.float32)
            stds = np.array([norm_info["std"][c] for c in feature_cols], dtype=np.float32)
            # 安全防护：若标准差极小（例如常数特征或极其微弱的方差特征，设置保护门限为 2e-2），将其保护性设为 1.0，防止除以极小值引起噪声极度放大或模型数值爆炸
            stds = np.where(stds < 2e-2, 1.0, stds)
            
            # 智能双模型加载：后段测距平滑 (stabilizer) 默认使用 best.pth，轨迹预测约束 (trajectory) 强制使用 gru_enhanced.pth
            stabilizer_checkpoint = torch.load(model_path, map_location="cpu")
            is_trajectory_net = False
            if isinstance(stabilizer_checkpoint, dict) and "model_config" in stabilizer_checkpoint:
                cfg = stabilizer_checkpoint["model_config"]
                if "future_steps" in cfg or cfg.get("input_dim") == 12:
                    is_trajectory_net = True
            elif isinstance(stabilizer_checkpoint, dict) and "model_state_dict" in stabilizer_checkpoint:
                s_dict = stabilizer_checkpoint["model_state_dict"]
                if "classifier.0.weight" in s_dict or "predictor.0.weight" in s_dict:
                    is_trajectory_net = True
            
            # 若用户在 UI 选择加载的是轨迹预测模型，后段测距平滑自动降级路由到 best.pth
            if is_trajectory_net:
                fallback_model_path = Path(model_path).parent / "best.pth"
                if not fallback_model_path.exists():
                    fallback_model_path = distance_model_dir / "models" / "best.pth"
                
                if fallback_model_path.exists():
                    stabilizer_checkpoint = torch.load(str(fallback_model_path), map_location="cpu")
                else:
                    print("--> [ERROR] 未能定位默认的 models/best.pth 文件！")
            
            model = GRUDistanceStabilizer(input_dim=input_dim)
            if isinstance(stabilizer_checkpoint, dict) and "model_state_dict" in stabilizer_checkpoint:
                model.load_state_dict(stabilizer_checkpoint["model_state_dict"])
            else:
                model.load_state_dict(stabilizer_checkpoint)
            model.eval()
            
            # 独立加载 UAVTrajectoryNet 用于分割轨迹真伪验证与未来偏移预测
            trajectory_model = None
            if is_pure_video and detector_mode == "图像分割+位移追踪":
                from segmentation_mode.model import UAVTrajectoryNet
                enhanced_path = distance_model_dir / "segmentation_mode" / "models" / "gru_enhanced.pth"
                if enhanced_path.exists():
                    print(f"--> [GUI] 成功加载智能时序轨迹预测模型: {enhanced_path.name}")
                    trajectory_model = UAVTrajectoryNet.load_from_checkpoint(str(enhanced_path), device="cpu")
                else:
                    print(f"--> [WARN] 未能定位轨迹预测模型 {enhanced_path}，已降级为不进行时序 GRU 轨迹分类过滤！")
            
            stable_predictions_dict = {}
            if not is_pure_video:
                seq_len = 25
                # 分点位段独立推理以防止点位间隙引入突变
                unique_pts = df_pt["point_id"].dropna().unique()
                for pt in unique_pts:
                    df_single_pt = df_pt[df_pt["point_id"] == pt].sort_values("global_frame_id")
                    n_pt_frames = len(df_single_pt)
                    pt_preds = np.zeros(n_pt_frames, dtype=np.float32)
                    raw_preds = df_single_pt["raw_predicted_range_m"].to_numpy(dtype=np.float32)
                    pt_preds[:seq_len-1] = raw_preds[:seq_len-1]
                    
                    for i in range(seq_len - 1, n_pt_frames):
                        sub_df = df_single_pt.iloc[i - seq_len + 1 : i + 1]
                        x_raw = sub_df[feature_cols].to_numpy(dtype=np.float32)
                        x_norm = (x_raw - means) / stds
                        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            last_pred, _ = model(x_t)
                        pt_preds[i] = last_pred.item()
                        
                    # 存入字典映射，使用全局帧 global_frame_id 作为 Key
                    base_idx = df_single_pt.index[0]
                    for idx, row in df_single_pt.iterrows():
                        stable_predictions_dict[int(row["global_frame_id"])] = pt_preds[idx - base_idx]
                        
                print("   模型预测完毕，成功生成平滑测距时序曲线。")
            
            # 4. 初始化合成图像与视频参数
            print("4. 初始化视频合成画布...")
            video_w, video_h = 960, 540
            chart_w, chart_h = 720, 540
            out_w, out_h = video_w + chart_w, video_h
            
            dpi_val = self.dpi_var.get()
            if dpi_val <= 0:
                dpi_val = 100
            fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=dpi_val)
            
            user_output_val = self.output_dir_var.get().strip()
            if "outputs (自动与输入物理路径严格对齐)" in user_output_val or not user_output_val:
                base_reports_dir = Path("F:/UAV Trajectory System/distance model/outputs/reports")
            else:
                base_reports_dir = Path(user_output_val)
            
            base_reports_dir.mkdir(parents=True, exist_ok=True)
            
            # 自动递增寻找 test00x 目录
            existing_tests = []
            if base_reports_dir.exists():
                for item in base_reports_dir.iterdir():
                    if item.is_dir() and item.name.startswith("test") and item.name[4:].isdigit():
                        try:
                            existing_tests.append(int(item.name[4:]))
                        except ValueError:
                            pass
            next_idx = 1
            if existing_tests:
                next_idx = max(existing_tests) + 1
            
            test_folder_name = f"test{next_idx:03d}"
            output_dir = base_reports_dir / test_folder_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if is_pure_video:
                file_tag = "pure_video_yolov5" if detector_mode == "YOLOv5 实时目标框定" else "pure_video_segmentation"
                video_name = f"visualized_{file_tag}.mp4"
            else:
                if is_session_mode:
                    video_name = f"visualized_distance_session_{session_id}.mp4"
                else:
                    video_name = f"visualized_distance_point_{point_id}.mp4"
            
            self.output_video_path = str(output_dir / video_name)
            
            fps_val = self.fps_var.get()
            if fps_val <= 0:
                fps_val = 25.0
                
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(self.output_video_path, fourcc, fps_val, (out_w, out_h))
            
            # 将 reports_dir 直接设为 test00x，所有输出（CSV, events, 快照）都归拢至该文件夹
            self.reports_dir = output_dir
            reports_dir = self.reports_dir
            
            if video_file and video_file.exists():
                cap = cv2.VideoCapture(str(video_file))
            else:
                cap = None
            
            # 运行端状态与时序组件初始化 (P4 & P5 & P7 阶段)
            from src.uav_distance_pipeline.runtime_motion_gate import RuntimeMotionGate
            
            motion_gate = RuntimeMotionGate(run_cfg)
            self.reset_gru_state("INITIALIZATION")
            
            # 初始化卡尔曼与状态机状态
            tracking_state = "LOST"
            consecutive_valid_frames = 0
            coasting_start_time = None
            last_target_center = None
            last_valid_bbox = None
            
            # 引入智能特征轨迹多目标追踪器
            tracker = FeatureTracker(seq_len=20, input_dim=12)
            
            # 保存各阶测距历史用于示波器和 CSV 导出
            history_frames = []
            history_robust_physics = []
            history_stable = []
            history_time = []
            
            frame_logs = []
            prev_msec = 0.0
            
            snapshot_frames = [int(total_render_frames * 0.2), int(total_render_frames * 0.5), int(total_render_frames * 0.8)]
            
            print("5. 正在渲染拼接帧，并评估卡尔曼运动门控与状态机...")
            
            # 5. 循环合成每一帧
            for i in range(total_render_frames):
                current_frame_id = start_frame + i
                
                if cap is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_id)
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
                    
                orig_h, orig_w = frame.shape[:2]
                frame_resized = cv2.resize(frame, (video_w, video_h))
                
                scale_x = video_w / orig_w
                scale_y = video_h / orig_h
                
                # 初始化单帧变量
                measurement_valid = False
                invalid_reason = ""
                bw = 0.0
                bh = 0.0
                area_px = 0.0
                aspect_ratio = 1.0
                center_u = 1280.0
                center_v = 720.0
                center_u_norm = 0.5
                center_v_norm = 0.5
                conf = 0.0
                
                # OBB 特征
                obb_long_axis_px = None
                obb_short_axis_px = None
                obb_angle_deg = None
                obb_angle_sin = None
                obb_angle_cos = None
                mask_area_px = None
                convex_hull_area_px = None
                solidity = None
                component_count = 1
                contour_perimeter = None
                eccentricity = None
                
                box_color = (0, 255, 255) # 默认黄色
                
                # 计算真实时间戳和真实时间差 dt (P9.1 阶段)
                if cap is not None:
                    curr_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    curr_time_s = curr_msec / 1000.0
                    if i == 0:
                        dt = 1.0 / fps_val
                    else:
                        dt = curr_time_s - (prev_msec / 1000.0)
                        if dt <= 0:
                            dt = 1.0 / fps_val
                            self.log_event("TIME_GAP_DETECTED", {"frame_id": current_frame_id, "dt": dt})
                        elif dt > 2.0:
                            self.log_event("TIME_GAP_DETECTED", {"frame_id": current_frame_id, "dt": dt})
                            dt = 1.0 / fps_val
                    prev_msec = curr_msec
                else:
                    curr_time_s = i / fps_val
                    dt = 1.0 / fps_val
                
                actual_fps = 1.0 / dt if dt > 0.0 else fps_val
                
                # 根据不同模式做目标提取
                if is_pure_video:
                    if detector_mode == "YOLOv5 实时目标框定":
                        pred = detector.detect(frame)
                        if len(pred) > 0:
                            pred = pred[np.argsort(pred[:, 4])[::-1]]
                            x1_orig, y1_orig, x2_orig, y2_orig, conf_val = pred[0, 0], pred[0, 1], pred[0, 2], pred[0, 3], pred[0, 4]
                            bw_val = x2_orig - x1_orig
                            bh_val = y2_orig - y1_orig
                            
                            # 极小像素过滤
                            if bw_val < run_cfg.get("measurement_quality", {}).get("min_bbox_width_px", 3.0) or bh_val < run_cfg.get("measurement_quality", {}).get("min_bbox_height_px", 3.0):
                                measurement_valid = False
                                invalid_reason = "BBOX_TOO_SMALL"
                            else:
                                bw = bw_val
                                bh = bh_val
                                area_px = bw * bh
                                aspect_ratio = bw / bh
                                center_u = (x1_orig + x2_orig) / 2.0
                                center_v = (y1_orig + y2_orig) / 2.0
                                center_u_norm = center_u / 2560.0
                                center_v_norm = center_v / 1440.0
                                conf = conf_val
                                measurement_valid = True
                                last_valid_bbox = (x1_orig, y1_orig, x2_orig, y2_orig, conf)
                        else:
                            measurement_valid = False
                            invalid_reason = "TARGET_MISSING"
                            
                    else:
                        # ===== 图像分割检测追踪器模式 (深度重构版) =====
                        # 1. 驱动多目标智能特征追踪器，执行多域融合、贪婪匹配、去重及背景死锁阻断
                        tracker.update(frame, csv_row=None)
                        
                        # 2. 对当前所有活跃的 tracks 运行时序特征提取与 UAV 真伪分类预测
                        for tid, track in tracker.tracks.items():
                            if len(track.history_buffer) == tracker.seq_len:
                                features_np = tracker.get_track_features(track)
                                features_t = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0)
                                
                                if trajectory_model is not None:
                                    with torch.no_grad():
                                        logits_uav, pred_offsets = trajectory_model(features_t)
                                        prob = torch.sigmoid(logits_uav).item()
                                        pred_offsets = pred_offsets.squeeze(0).cpu().numpy()
                                        
                                    # 一阶指数平滑轨迹概率，平抑运动学跳变与分割缺陷带来的分类分类概率抖动
                                    if getattr(track, "smooth_classification_prob", None) is None:
                                        track.smooth_classification_prob = prob
                                    else:
                                        track.smooth_classification_prob = 0.7 * track.smooth_classification_prob + 0.3 * prob
                                        
                                    track.classification_prob = track.smooth_classification_prob
                                    track.is_uav = (track.smooth_classification_prob >= 0.55) # 平滑判定门限
                                    
                                    # 提取未来 5 帧的轨迹偏移量，乘上画幅比例尺度，获得当前像素分辨率下的预测坐标
                                    current_x, current_y = track.smooth_px, track.smooth_py
                                    scale_x_px = 2560.0 / 2.0
                                    scale_y_px = 1440.0 / 2.0
                                    track_pred = []
                                    for off in pred_offsets:
                                        track_pred.append((int(current_x + off[0] * scale_x_px), int(current_y + off[1] * scale_y_px)))
                                    track.pred_coords = track_pred
                                else:
                                    track.classification_prob = 0.80
                                    track.is_uav = True
                                    track.pred_coords = []
                            else:
                                track.classification_prob = None
                                track.is_uav = False
                                track.pred_coords = []
                                
                        # 3. 甄选高置信度的主黄金轨迹，严格过滤时序确诊的鸟等背景噪声干扰
                        primary_track = None
                        uav_tracks = [t for t in tracker.tracks.values() if t.is_uav]
                        candidate_tracks = [
                            t for t in tracker.tracks.values()
                            if t.is_uav or (len(t.history_buffer) < tracker.seq_len)
                        ]
                        
                        if uav_tracks:
                            primary_track = max(uav_tracks, key=lambda t: t.classification_prob)
                        elif candidate_tracks:
                            # 降级选择预热中的潜在候选目标（排除了已被分类器确诊的噪声）
                            primary_track = min(candidate_tracks, key=lambda t: t.track_id)
                            
                        # 4. 主黄金轨迹特征传导以供后段测距稳定器计算
                        if primary_track is not None and primary_track.history_buffer:
                            # 获取一阶滑动指数滤波后的防抖中心点和宽高
                            center_u = primary_track.smooth_px
                            center_v = primary_track.smooth_py
                            bw = max(2.0, primary_track.smooth_w)
                            bh = max(2.0, primary_track.smooth_h)
                            
                            x1_orig = max(0.0, min(2560.0, center_u - bw / 2.0))
                            y1_orig = max(0.0, min(1440.0, center_v - bh / 2.0))
                            x2_orig = max(0.0, min(2560.0, center_u + bw / 2.0))
                            y2_orig = max(0.0, min(1440.0, center_v + bh / 2.0))
                            
                            area_px = bw * bh
                            aspect_ratio = bw / bh
                            center_u_norm = center_u / 2560.0
                            center_v_norm = center_v / 1440.0
                            conf = primary_track.classification_prob if primary_track.classification_prob is not None else 0.80
                            measurement_valid = True
                            
                            # 提取 OBB 与凸包几何参数 (在原始 contours 中做临近索引)
                            obb_long_axis_px = bw
                            obb_short_axis_px = bh
                            obb_angle_deg = 0.0
                            solidity = 0.85
                            contour_perimeter = 2.0 * (bw + bh)
                            
                            for contour_item in tracker.last_contours:
                                rx_c, ry_c, rw_c, rh_c = cv2.boundingRect(contour_item)
                                cx_c = rx_c + rw_c / 2.0
                                cy_c = ry_c + rh_c / 2.0
                                if np.hypot(cx_c - center_u, cy_c - center_v) < 30.0:
                                    rect = cv2.minAreaRect(contour_item)
                                    (cx_obb, cy_obb), (w_obb, h_obb), angle_obb = rect
                                    obb_long_axis_px = max(2.0, max(w_obb, h_obb))
                                    obb_short_axis_px = max(2.0, min(w_obb, h_obb))
                                    obb_angle_deg = angle_obb
                                    
                                    hull = cv2.convexHull(contour_item)
                                    hull_area = cv2.contourArea(hull)
                                    solidity = (rw_c * rh_c) / hull_area if hull_area > 0 else 0.85
                                    contour_perimeter = cv2.arcLength(contour_item, True)
                                    break
                                    
                            obb_angle_sin = float(np.sin(np.radians(obb_angle_deg)))
                            obb_angle_cos = float(np.cos(np.radians(obb_angle_deg)))
                            mask_area_px = area_px
                            convex_hull_area_px = area_px / solidity
                            eccentricity = float(np.sqrt(1.0 - (obb_short_axis_px / max(1e-6, obb_long_axis_px))**2))
                            
                            last_target_center = (center_u, center_v)
                            last_valid_bbox = (x1_orig, y1_orig, x2_orig, y2_orig, conf)
                            
                            # 5. 可视化绘制：在原画幅中用绿色实心小圆点和连接线画出未来 5 帧偏移轨迹
                            if primary_track.is_uav and primary_track.pred_coords:
                                for pt_idx, coord in enumerate(primary_track.pred_coords):
                                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 6, (0, 255, 0), -1, cv2.LINE_AA)
                                    if pt_idx > 0:
                                        cv2.line(frame, (int(primary_track.pred_coords[pt_idx-1][0]), int(primary_track.pred_coords[pt_idx-1][1])),
                                                 (int(coord[0]), int(coord[1])), (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            measurement_valid = False
                            invalid_reason = "TARGET_MISSING"
                else:
                    # 从 df_pt 中检索匹配的帧记录
                    row_matches = df_pt[df_pt["global_frame_id"] == current_frame_id]
                    if not row_matches.empty:
                        row = row_matches.iloc[0]
                        is_missing = int(row["is_missing"])
                        is_interpolated = int(row["is_interpolated"])
                        
                        if is_missing == 0:
                            bw = float(row["bbox_width_px"])
                            bh = float(row["bbox_height_px"])
                            area_px = float(row["bbox_area_px"])
                            aspect_ratio = float(row["bbox_aspect_ratio"])
                            center_u = float(row["bbox_center_u"])
                            center_v = float(row["bbox_center_v"])
                            center_u_norm = float(row["bbox_center_u_norm"])
                            center_v_norm = float(row["bbox_center_v_norm"])
                            conf = float(row["detector_confidence"])
                            measurement_valid = True
                            
                            x1_orig = float(row["bbox_x1"])
                            y1_orig = float(row["bbox_y1"])
                            x2_orig = float(row["bbox_x2"])
                            y2_orig = float(row["bbox_y2"])
                            last_valid_bbox = (x1_orig, y1_orig, x2_orig, y2_orig, conf)
                        else:
                            measurement_valid = False
                            invalid_reason = "TARGET_MISSING"
                            
                # 物理距离估计与中位偏离稳健融合 (P2 & P5 & P6 阶段)
                raw_physics_range = 0.0
                robust_physics_range = 0.0
                q_candidates = [0.0]
                
                if measurement_valid:
                    # 校验短轴塌陷
                    if bh <= 1.0 or aspect_ratio >= 8.0:
                        measurement_valid = False
                        invalid_reason = "SHORT_AXIS_COLLAPSED"
                        self.log_event("SHORT_AXIS_COLLAPSED", {"frame_id": current_frame_id, "bw": bw, "bh": bh})
                        
                if measurement_valid:
                    d_width = 11494.25 * 0.30 / max(1.0, bw)
                    d_height = 11494.25 * 0.15 / max(1.0, bh)
                    d_area = np.sqrt(11494.25 * 11494.25 * 0.30 * 0.15 / max(1.0, area_px))
                    
                    raw_physics_range = (d_width + d_height + d_area) / 3.0
                    
                    # 稳健中位数融合
                    d_candidates = [d_width, d_height, d_area]
                    q_candidates = [1.0, 1.0, 1.0]
                    
                    if obb_long_axis_px is not None:
                        d_obb = 11494.25 * 0.30 / max(1.0, obb_long_axis_px)
                        d_candidates.append(d_obb)
                        q_candidates.append(1.0)
                        
                    # 偏离中位数过大过滤
                    med_val = np.median(d_candidates)
                    disagreement_thresh = run_cfg.get("measurement_quality", {}).get("max_physics_disagreement_ratio", 0.5)
                    for idx, d_val in enumerate(d_candidates):
                        if abs(d_val - med_val) / max(1.0, med_val) > disagreement_thresh:
                            q_candidates[idx] *= 0.1
                            self.log_event("PHYSICS_DISAGREEMENT", {"frame_id": current_frame_id, "d_val": d_val, "med_val": med_val})
                            
                    # solidity 质量判定 (大目标卡紧，小目标自适应放宽以防止漏检)
                    current_min_solidity = 0.80
                    if area_px < 35.0:
                        current_min_solidity = 0.50
                    elif area_px < 100.0:
                        current_min_solidity = 0.70
                        
                    if solidity is not None and solidity < current_min_solidity:
                        for idx in range(len(q_candidates)):
                            q_candidates[idx] *= 0.5
                        self.log_event("MASK_FRAGMENTED", {"frame_id": current_frame_id, "solidity": solidity})
                        
                    sum_q = sum(q_candidates)
                    if sum_q < 1.5:
                        measurement_valid = False
                        invalid_reason = "PHYSICS_DISAGREEMENT"
                        robust_physics_range = raw_physics_range
                    else:
                        robust_physics_range = sum(q * d for q, d in zip(q_candidates, d_candidates)) / sum_q
                else:
                    # 无效观测时的物理 fallback
                    if last_valid_bbox is not None:
                        x1_orig, y1_orig, x2_orig, y2_orig, _ = last_valid_bbox
                        bw_fallback = x2_orig - x1_orig
                        bh_fallback = y2_orig - y1_orig
                        area_fallback = bw_fallback * bh_fallback
                    else:
                        bw_fallback = 23.36327
                        bh_fallback = 11.63500
                        area_fallback = 611.3737
                    d_w_f = 11494.25 * 0.30 / max(1.0, bw_fallback)
                    d_h_f = 11494.25 * 0.15 / max(1.0, bh_fallback)
                    d_a_f = np.sqrt(11494.25 * 11494.25 * 0.30 * 0.15 / max(1.0, area_fallback))
                    robust_physics_range = (d_w_f + d_h_f + d_a_f) / 3.0
                    raw_physics_range = robust_physics_range
                    
                # 针对最远 500m 测距场景实施高位硬限幅防护，消除小尺寸噪声引起的高大跳跃
                robust_physics_range = np.clip(robust_physics_range, 0.0, 500.0)
                raw_physics_range = np.clip(raw_physics_range, 0.0, 500.0)
                
                # 4. 卡尔曼自适应更新与物理运动门控 (P7 阶段)
                gate_decision = "ACCEPT"
                kalman_dist = robust_physics_range
                radial_vel = 0.0
                
                if measurement_valid:
                    if tracking_state == "LOST":
                        # LOST 状态下直接接受，不执行创新值门控，以便在连续帧后顺利确认重捕获
                        gate_decision = "ACCEPT"
                        kalman_dist = robust_physics_range
                        radial_vel = 0.0
                    else:
                        gate_decision, kalman_dist, radial_vel = motion_gate.process(robust_physics_range, curr_time_s, measurement_valid=True)
                        if gate_decision == "REJECT":
                            measurement_valid = False
                            invalid_reason = "TRACK_MISMATCH"
                            self.log_event("MEASUREMENT_REJECTED", {"frame_id": current_frame_id, "measured": robust_physics_range, "predicted": kalman_dist})
                else:
                    # Coasting 推理更新（无有效观测），仅做卡尔曼预测，不使用 fallback 虚拟观测污染状态
                    gate_decision, kalman_dist, radial_vel = motion_gate.process(robust_physics_range, curr_time_s, measurement_valid=False)
                    
                # 5. 目标跟踪状态机更新 (P4 阶段)
                prev_state = tracking_state
                
                if tracking_state == "LOST":
                    if measurement_valid:
                        consecutive_valid_frames += 1
                        if consecutive_valid_frames >= run_cfg.get("tracking", {}).get("reacquire_confirm_frames", 3):
                            tracking_state = "REACQUIRING"
                            self.reset_gru_state("TARGET_REACQUIRED")
                            motion_gate.reset(robust_physics_range)
                            consecutive_valid_frames = 0
                            tracking_state = "WARMING_UP"
                    else:
                        consecutive_valid_frames = 0
                        last_target_center = None
                        last_valid_bbox = None
                        tracker.tracks.clear()
                        
                elif tracking_state == "WARMING_UP":
                    if measurement_valid:
                        consecutive_valid_frames += 1
                        if consecutive_valid_frames >= run_cfg.get("tracking", {}).get("gru_warmup_frames", 25):
                            tracking_state = "TRACKING"
                    else:
                        consecutive_valid_frames = 0
                        tracking_state = "COASTING"
                        coasting_start_time = curr_time_s
                        
                elif tracking_state in ["TRACKING", "DEGRADED"]:
                    if measurement_valid:
                        if gate_decision == "DOWNWEIGHT":
                            tracking_state = "DEGRADED"
                        else:
                            tracking_state = "TRACKING"
                    else:
                        tracking_state = "COASTING"
                        coasting_start_time = curr_time_s
                        
                elif tracking_state == "COASTING":
                    if measurement_valid:
                        consecutive_valid_frames += 1
                        if consecutive_valid_frames >= run_cfg.get("tracking", {}).get("reacquire_confirm_frames", 3):
                            tracking_state = "TRACKING"
                    else:
                        consecutive_valid_frames = 0
                        # 检查超时
                        coast_elapsed = curr_time_s - coasting_start_time
                        if coast_elapsed > run_cfg.get("tracking", {}).get("lost_timeout_s", 1.0):
                            tracking_state = "LOST"
                            self.reset_gru_state("TARGET_LOST")
                            last_target_center = None
                            last_valid_bbox = None
                            tracker.tracks.clear()
                            
                # 状态机改变日志
                if tracking_state != prev_state:
                    self.log_event("TRACK_COMPONENT_SWITCH", {
                        "from": prev_state,
                        "to": tracking_state,
                        "frame_id": current_frame_id
                    })
                    if tracking_state == "LOST":
                        self.log_event("TARGET_LOST", {"frame_id": current_frame_id})
                    elif tracking_state == "TRACKING" and prev_state == "WARMING_UP":
                        self.log_event("TARGET_REACQUIRED", {"frame_id": current_frame_id})
                        
                # 6. MLP 和 GRU 推理段 (P3 & P10 阶段)
                mlp_predicted_residual_m = 0.0
                raw_predicted_range_m = robust_physics_range
                stable_pred_val = robust_physics_range
                
                # 仅在非 LOST 状态下执行级联推理
                if tracking_state != "LOST":
                    feat_dict = {
                        "bbox_x1": x1_orig,
                        "bbox_y1": y1_orig,
                        "bbox_x2": x2_orig,
                        "bbox_y2": y2_orig,
                        "bbox_width_px": bw,
                        "bbox_height_px": bh,
                        "bbox_area_px": area_px,
                        "bbox_aspect_ratio": aspect_ratio,
                        "bbox_center_u": center_u,
                        "bbox_center_v": center_v,
                        "bbox_center_u_norm": center_u_norm,
                        "bbox_center_v_norm": center_v_norm,
                        "detector_confidence": conf,
                        "is_missing": float(not measurement_valid),
                        "is_interpolated": 0.0,
                        "physics_width_range_m": 11494.25 * 0.30 / max(1.0, bw),
                        "physics_height_range_m": 11494.25 * 0.15 / max(1.0, bh),
                        "physics_area_range_m": np.sqrt(11494.25 * 11494.25 * 0.30 * 0.15 / max(1.0, area_px)),
                        "fused_range_m": robust_physics_range,
                        "frame_delta_t_s": dt
                    }
                    
                    # MLP 预测
                    mlp_x_raw = np.array([feat_dict[col] for col in mlp_feature_cols], dtype=np.float32).reshape(1, -1)
                    mlp_predicted_residual_m = predict_mlp(mlp_params, mlp_x_raw)[0]
                    
                    if abs(mlp_predicted_residual_m) > 10.0:
                        self.log_event("MLP_RESIDUAL_OUTLIER", {"frame_id": current_frame_id, "residual": mlp_predicted_residual_m})
                        
                    raw_predicted_range_m = robust_physics_range + mlp_predicted_residual_m
                    # 物理下限保护：若神经网络 MLP 外推出负值或 0 值，则安全降级回 robust_physics_range 物理测距值
                    if raw_predicted_range_m <= 0.0:
                        raw_predicted_range_m = robust_physics_range
                        
                    feat_dict["mlp_predicted_residual_m"] = mlp_predicted_residual_m
                    feat_dict["raw_predicted_range_m"] = raw_predicted_range_m
                    
                    # GRU 时序特征
                    gru_feat_vector = np.array([feat_dict[col] for col in feature_cols], dtype=np.float32)
                    self.features_queue.append(gru_feat_vector)
                    if len(self.features_queue) > 25:
                        self.features_queue.pop(0)
                        
                    if tracking_state == "WARMING_UP" or len(self.features_queue) < 25:
                        stable_pred_val = raw_predicted_range_m
                    else:
                        x_raw_gru = np.array(self.features_queue, dtype=np.float32)
                        x_norm_gru = (x_raw_gru - means) / stds
                        x_t = torch.tensor(x_norm_gru, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            last_pred, _ = model(x_t)
                        stable_pred_val = last_pred.item()
                        
                    # 统一负距离拦截、NaN异常与物理合理性门控处理
                    # 拦截所有负距离、NaN异常，或与卡尔曼物理滤波器偏差超过 30 米的异常值（标志神经网络发生溢出/严重幻觉），安全降级至卡尔曼物理测距值
                    if (stable_pred_val <= 0.0 or 
                        np.isnan(stable_pred_val) or 
                        abs(stable_pred_val - kalman_dist) > 30.0):
                        
                        self.log_event("GRU_OUTPUT_INVALID", {
                            "frame_id": current_frame_id, 
                            "val": stable_pred_val, 
                            "kalman": kalman_dist
                        })
                        stable_pred_val = kalman_dist
                        
                    # 针对 500m 内实测场景实施最终输出硬门限卡幅拦截，剔除残差累加溢出导致的距离飞越
                    if stable_pred_val > 500.0:
                        stable_pred_val = 500.0
                else:
                    stable_pred_val = float('nan')
                    raw_predicted_range_m = float('nan')
                    
                # 7. 绘图与绘制 HUD 面板展示 (P8 阶段)
                if measurement_valid and tracking_state != "LOST":
                    x1 = int(x1_orig * scale_x)
                    y1 = int(y1_orig * scale_y)
                    x2 = int(x2_orig * scale_x)
                    y2 = int(y2_orig * scale_y)
                    if tracking_state == "TRACKING":
                        status_color = (0, 255, 0)
                    elif tracking_state in ["DEGRADED", "COASTING", "WARMING_UP"]:
                        status_color = (0, 255, 255)
                    else:
                        status_color = (0, 0, 255)
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), status_color, 2)
                    
                    if not np.isnan(stable_pred_val):
                        cv2.putText(frame_resized, f"Est: {stable_pred_val:.2f}m", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)
                
                # 绘制 HUD 半透明背板
                overlay = frame_resized.copy()
                cv2.rectangle(overlay, (15, 15), (320, 275), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.6, frame_resized, 0.4, 0, frame_resized)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                hud_id = session_id if is_session_mode else point_id
                if is_pure_video:
                    detector_tag = "YOLOv5" if detector_mode == "YOLOv5 实时目标框定" else "Segmentation"
                    hud_id = f"[Pure Video mode - {detector_tag}]"
                
                if tracking_state == "TRACKING":
                    text_color = (0, 255, 0)
                elif tracking_state in ["DEGRADED", "COASTING", "WARMING_UP"]:
                    text_color = (0, 255, 255)
                else:
                    text_color = (0, 0, 255)
                    
                cv2.putText(frame_resized, f"ID: {hud_id}", (25, 33), font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Time: {curr_time_s:.2f}s | Frame: {current_frame_id}", (25, 52), font, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Tracking State: {tracking_state}", (25, 71), font, 0.40, text_color, 1, cv2.LINE_AA)
                
                raw_phy_str = f"{raw_physics_range:.2f} m" if tracking_state != "LOST" else "N/A"
                rob_phy_str = f"{robust_physics_range:.2f} m" if tracking_state != "LOST" else "N/A"
                mlp_str = f"{mlp_predicted_residual_m:.3f} m" if tracking_state != "LOST" else "N/A"
                
                if tracking_state == "LOST":
                    gru_str = "N/A"
                    final_str = "N/A"
                elif tracking_state == "WARMING_UP":
                    gru_str = f"WARMING_UP {len(self.features_queue)}/25"
                    final_str = f"{stable_pred_val:.2f} m"
                else:
                    gru_str = f"{stable_pred_val:.2f} m"
                    final_str = f"{stable_pred_val:.2f} m"
                    
                quality_val = float(sum(q_candidates) / len(q_candidates)) if measurement_valid else 0.0
                
                cv2.putText(frame_resized, f"Raw Physics: {raw_phy_str}", (25, 93), font, 0.40, (255, 120, 0), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Robust Physics: {rob_phy_str}", (25, 112), font, 0.40, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Residual MLP: {mlp_str}", (25, 131), font, 0.40, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"GRU: {gru_str}", (25, 150), font, 0.40, (255, 192, 203), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Final Stable: {final_str}", (25, 170), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Measurement Quality: {quality_val:.2f}", (25, 190), font, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Actual FPS: {actual_fps:.2f}", (25, 210), font, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Frame Delta: {dt:.4f}s", (25, 230), font, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame_resized, f"Invalid Reason: {invalid_reason}", (25, 250), font, 0.40, (0, 0, 255) if invalid_reason else (128, 128, 128), 1, cv2.LINE_AA)
                
                if tracking_state == "LOST":
                    cv2.putText(frame_resized, "[LOST]", (20, video_h - 20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                elif tracking_state == "COASTING":
                    cv2.putText(frame_resized, "[COASTING]", (20, video_h - 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                elif tracking_state == "WARMING_UP":
                    cv2.putText(frame_resized, "[WARMING_UP]", (20, video_h - 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                
                # 记录帧级诊断数据
                frame_logs.append({
                    "frame_id": current_frame_id,
                    "timestamp": curr_time_s,
                    "actual_fps": actual_fps,
                    "frame_delta_t_s": dt,
                    "tracking_state": tracking_state,
                    "measurement_valid": float(measurement_valid),
                    "measurement_quality": float(quality_val),
                    "invalid_reason": invalid_reason,
                    "bbox_width_px": bw,
                    "bbox_height_px": bh,
                    "bbox_area_px": area_px,
                    "obb_long_axis_px": obb_long_axis_px if obb_long_axis_px is not None else float('nan'),
                    "obb_short_axis_px": obb_short_axis_px if obb_short_axis_px is not None else float('nan'),
                    "mask_area_px": mask_area_px if mask_area_px is not None else float('nan'),
                    "solidity": solidity if solidity is not None else float('nan'),
                    "component_count": component_count,
                    "raw_physics_range_m": raw_physics_range,
                    "robust_physics_range_m": robust_physics_range,
                    "mlp_range_m": raw_predicted_range_m,
                    "kalman_range_m": kalman_dist,
                    "gru_range_m": stable_pred_val,
                    "final_stable_range_m": stable_pred_val if tracking_state != "LOST" else float('nan')
                })
                
                # 绘制 Matplotlib 右侧动态图表
                ax.clear()
                
                if is_pure_video:
                    if measurement_valid and tracking_state != "LOST":
                        history_frames.append(current_frame_id)
                        history_robust_physics.append(robust_physics_range)
                        history_stable.append(stable_pred_val)
                        history_time.append(curr_time_s)
                    
                    if history_time:
                        ax.plot(history_time, history_robust_physics, color="blue", linestyle="--", alpha=0.6, label="Physics Fused")
                        clean_t = [t for t, v in zip(history_time, history_stable) if not np.isnan(v)]
                        clean_v = [v for v in history_stable if not np.isnan(v)]
                        if clean_v:
                            ax.plot(clean_t, clean_v, color="red", linestyle="-", linewidth=1.8, label="GRU Stabilizer")
                        
                        min_val = min(history_robust_physics)
                        max_val = max(history_robust_physics)
                        val_span = max_val - min_val
                        if val_span < 1.0:
                            val_span = 10.0
                        y_limits = (min_val - 0.15 * val_span - 5.0, max_val + 0.15 * val_span + 5.0)
                        
                        if measurement_valid and tracking_state != "LOST" and not np.isnan(stable_pred_val):
                            ax.plot([curr_time_s], [stable_pred_val], "ro")
                    else:
                        y_limits = (0, 100)
                else:
                    df_history = df_pt[(df_pt["global_frame_id"] <= current_frame_id) & (df_pt["is_missing"] == 0)].sort_values("global_frame_id")
                    
                    if not df_history.empty:
                        h_frames = df_history["global_frame_id"].to_numpy()
                        h_time_s = (h_frames - start_frame) / fps_val
                        h_true = df_history["true_range_m"].to_numpy()
                        h_fused = df_history["fused_range_m"].to_numpy()
                        h_stable = np.array([stable_predictions_dict.get(int(r["global_frame_id"]), float(r["raw_predicted_range_m"])) for fid, r in df_history.iterrows()])
                        
                        ax.plot(h_time_s, h_true, color="green", linestyle="-", linewidth=2.0, label="True Range")
                        ax.plot(h_time_s, h_fused, color="blue", linestyle="--", alpha=0.6, label="Physics Fused")
                        ax.plot(h_time_s, h_stable, color="red", linestyle="-", linewidth=1.8, label="GRU Stabilizer")
                        
                        if current_frame_id in stable_predictions_dict:
                            curr_t = (current_frame_id - start_frame) / fps_val
                            ax.plot([curr_t], [stable_predictions_dict[current_frame_id]], "ro")
                        
                        min_val = min(h_true.min(), h_fused.min())
                        max_val = max(h_true.max(), h_fused.max())
                        val_span = max_val - min_val
                        if val_span < 1.0:
                            val_span = 10.0
                        y_limits = (min_val - 0.15 * val_span - 5.0, max_val + 0.15 * val_span + 5.0)
                    else:
                        y_limits = (0, 100)
                        
                ax.set_title("测距曲线实时对比 (Oscilloscope)", fontsize=11, fontweight="bold", pad=8)
                ax.set_xlabel("相对时间 (秒)", fontsize=9)
                ax.set_ylabel("距离估算值 (米)", fontsize=9)
                ax.set_xlim(0, total_render_frames / fps_val)
                ax.set_ylim(y_limits)
                ax.grid(True, linestyle=":", alpha=0.5)
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(loc="upper right", fontsize=8)
                
                fig.tight_layout()
                fig.canvas.draw()
                
                chart_rgba = np.asarray(fig.canvas.buffer_rgba())
                chart_bgr = cv2.cvtColor(chart_rgba, cv2.COLOR_RGBA2BGR)
                chart_resized = cv2.resize(chart_bgr, (chart_w, chart_h))
                
                combined_canvas = np.hstack((frame_resized, chart_resized))
                out_video.write(combined_canvas)
                
                # 保存特定帧快照
                if i in snapshot_frames:
                    if is_pure_video:
                        file_tag = "pure_video_yolov5" if detector_mode == "YOLOv5 实时目标框定" else "pure_video_segmentation"
                    else:
                        file_tag = point_id
                    snapshot_path = reports_dir / f"distance_snapshot_{file_tag}_frame_{current_frame_id}.png"
                    cv2.imwrite(str(snapshot_path), combined_canvas)
                    print(f"   已保存快照: {snapshot_path.name}")
                
                # 更新进度条
                progress = int((i + 1) / total_render_frames * 100)
                self.root.after(0, lambda p=progress: self.update_progress(p))
                
            # 8. 后期传统滤波计算与诊断报告导出 (P9 & P10 阶段)
            print("6. 执行诊断指标离线滤波器（中值滤波、EMA 等）计算并导出报表...")
            
            # 填充历史滤波
            ema_series = np.zeros(len(frame_logs))
            curr_ema = None
            alpha = run_cfg.get("filters", {}).get("ema_alpha", 0.15)
            
            for idx in range(len(frame_logs)):
                val = frame_logs[idx]["gru_range_m"]
                if np.isnan(val):
                    ema_series[idx] = np.nan
                else:
                    if curr_ema is None:
                        curr_ema = val
                    else:
                        curr_ema = alpha * val + (1.0 - alpha) * curr_ema
                    ema_series[idx] = curr_ema
                    
            w5 = run_cfg.get("filters", {}).get("median_short_window", 5)
            w25 = run_cfg.get("filters", {}).get("median_long_window", 25)
            
            for idx in range(len(frame_logs)):
                frame_logs[idx]["ema_range_m"] = float(ema_series[idx])
                
                # Median 5
                start5 = max(0, idx - w5 + 1)
                sub_series5 = [frame_logs[k]["gru_range_m"] for k in range(start5, idx + 1) if not np.isnan(frame_logs[k]["gru_range_m"])]
                frame_logs[idx]["median5_range_m"] = float(np.median(sub_series5)) if sub_series5 else float('nan')
                
                # Median 25
                start25 = max(0, idx - w25 + 1)
                sub_series25 = [frame_logs[k]["gru_range_m"] for k in range(start25, idx + 1) if not np.isnan(frame_logs[k]["gru_range_m"])]
                frame_logs[idx]["median25_range_m"] = float(np.median(sub_series25)) if sub_series25 else float('nan')
            
            # 保存到 reports 诊断文件夹
            diag_dir = self.reports_dir
            diag_dir.mkdir(parents=True, exist_ok=True)
            
            df_diag = pd.DataFrame(frame_logs)
            diag_csv_path = diag_dir / "frame_level_comparison.csv"
            df_diag.to_csv(diag_csv_path, index=False)
            print(f"   已导出帧级诊断对比 CSV: {diag_csv_path.name}")
            
            # 清理
            if cap is not None:
                cap.release()
            out_video.release()
            plt.close(fig)
            
            print(f"成功: 合成视频已导出至 {self.output_video_path}")
            
            # 自动分析并生成 test_summary.md 报告
            self.generate_report_summary(df_diag, diag_dir, Path(self.output_video_path).name, detector_mode)
            
            self.processing_finished(True)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"异常: 视频合成失败: {str(e)}")
            self.processing_finished(False)

    def update_progress(self, val: int):
        self.progress_bar["value"] = val

    def processing_finished(self, success: bool):
        self.is_running = False
        
        def _update_ui():
            self.run_btn.configure(state="normal")
            if success:
                self.play_btn.configure(state="normal")
                messagebox.showinfo("完成", f"视频渲染与对比分析成功完成！\n合成文件已保存至:\n{self.output_video_path}")
            else:
                messagebox.showerror("失败", "视频生成过程发生错误，请查看控制台日志输出。")
                
        self.root.after(0, _update_ui)

    def generate_report_summary(self, df, output_dir, video_name, detector_mode):
        try:
            total_frames = len(df)
            valid_df = df[df["final_stable_range_m"].notna() & (df["final_stable_range_m"] > 0)]
            valid_frames = len(valid_df)
            detection_rate = (valid_frames / total_frames) * 100 if total_frames > 0 else 0.0
            
            # 统计 invalid_reason
            invalid_reasons = {}
            if "invalid_reason" in df.columns:
                reasons_clean = df["invalid_reason"].fillna("VALID").replace("", "VALID")
                invalid_reasons = reasons_clean.value_counts().to_dict()
                
            # 计算各项测距数据
            stats = {}
            for col in ["raw_physics_range_m", "robust_physics_range_m", "kalman_range_m", "final_stable_range_m"]:
                if col in df.columns:
                    col_valid = df[df[col].notna() & (df[col] > 0)][col].values
                    if len(col_valid) > 0:
                        jumps = np.abs(np.diff(col_valid))
                        stats[col] = {
                            "count": int(len(col_valid)),
                            "min": float(np.min(col_valid)),
                            "max": float(np.max(col_valid)),
                            "mean": float(np.mean(col_valid)),
                            "std": float(np.std(col_valid)),
                            "mean_jump": float(np.mean(jumps)) if len(jumps) > 0 else 0.0,
                            "max_jump": float(np.max(jumps)) if len(jumps) > 0 else 0.0,
                            "clamped": int(np.sum(col_valid >= 499.99))
                        }
                    else:
                        stats[col] = {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "mean_jump": 0.0, "max_jump": 0.0, "clamped": 0}
            
            suggestions = []
            coasting_frames = 0
            tracking_frames = 0
            if "tracking_state" in df.columns:
                state_counts = df["tracking_state"].value_counts().to_dict()
                coasting_frames = state_counts.get("COASTING", 0)
                tracking_frames = state_counts.get("TRACKING", 0)
                
            missing_count = invalid_reasons.get("TARGET_MISSING", 0)
            mismatch_count = invalid_reasons.get("TRACK_MISMATCH", 0)
            
            if missing_count / total_frames > 0.3:
                suggestions.append("- **优化漏检抑制**：当前目标丢失率较高（{:.2f}%）。建议在图像分割阶段优化目标提取的二值化阈值或调小形态学滤波核，使得极远距离下微弱的像素斑点不被过滤。".format(missing_count/total_frames*100))
            if mismatch_count / total_frames > 0.1:
                suggestions.append("- **改进假阳性过滤**：有 {:.2f}% 的帧发生了轮廓错配（TRACK_MISMATCH）。建议调优小目标下的 Solidity 限制或引入更严格的纵横比过滤，减少非无人机斑点（例如云影、镜头微粒）的错配。".format(mismatch_count/total_frames*100))
            if stats.get("final_stable_range_m", {}).get("mean_jump", 0.0) > 5.0:
                suggestions.append("- **增强时序平滑度**：当前的稳定测距平均跳变（{:.2f}米）仍然偏高。可以考虑在 GRU 模型中增加阻尼权重，或者调整卡尔曼滤波的观测噪声协方差矩阵 $R$，增强滤波的平滑效果。".format(stats["final_stable_range_m"]["mean_jump"]))
            else:
                suggestions.append("- **保持当前滤波配置**：当前稳定测距的平均帧间跳变控制在 {:.2f} 米内，平滑性表现优异，可满足平稳输入要求。".format(stats.get("final_stable_range_m", {}).get("mean_jump", 0.0)))
            
            if stats.get("final_stable_range_m", {}).get("clamped", 0) / max(1, valid_frames) > 0.5:
                suggestions.append("- **增加超远测距外推上限**：目前有较大比例的有效帧（{:.1f}%）被限幅在 500m 阈值处。如果未来测试需要扩大范围，需要收集并训练 500m 以上的视频，并相应提高硬限幅门限（例如至 800m）。".format(stats["final_stable_range_m"]["clamped"]/valid_frames*100))
            
            # 写入 markdown 文件
            summary_path = output_dir / "test_summary.md"
            with open(summary_path, "w", encoding="utf-8") as sf:
                sf.write("# 单次测试评估报告（自动生成）\n\n")
                sf.write("## 1. 测试基本信息\n")
                sf.write("- **测试视频**: {}\n".format(video_name))
                sf.write("- **检测模式**: {}\n".format(detector_mode))
                sf.write("- **总处理帧数**: {} 帧\n".format(total_frames))
                sf.write("- **有效测距帧数**: {} 帧\n".format(valid_frames))
                sf.write("- **有效测距输出率**: {:.2f}%\n\n".format(detection_rate))
                
                sf.write("## 2. 测距平滑度对比\n\n")
                sf.write("| 数据源 | 有效帧数 | 范围 (m) | 均值 (m) | 标准差 (m) | 平均跳变 (m) | 最大跳变 (m) | 500m卡幅数 |\n")
                sf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
                for col, name in [
                    ("raw_physics_range_m", "原始物理距离"),
                    ("robust_physics_range_m", "物理防抖限制"),
                    ("kalman_range_m", "卡尔曼滤波"),
                    ("final_stable_range_m", "时序GRU滤波")
                ]:
                    st = stats.get(col, {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "mean_jump": 0.0, "max_jump": 0.0, "clamped": 0})
                    sf.write("| **{}** | {} | [{:.2f}, {:.2f}] | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {} |\n".format(
                        name, st["count"], st["min"], st["max"], st["mean"], st["std"], st["mean_jump"], st["max_jump"], st["clamped"]
                    ))
                sf.write("\n")
                
                sf.write("## 3. 检测状态与错配原因分布\n\n")
                if "invalid_reason" in df.columns:
                    sf.write("### 3.1 分割算法失效分析\n")
                    for reason, count in invalid_reasons.items():
                        sf.write("- **{}**: {} 帧 ({:.2f}%)\n".format(reason, count, count/total_frames*100))
                    sf.write("\n")
                
                if "tracking_state" in df.columns:
                    sf.write("### 3.2 追踪器航迹状态分布\n")
                    state_counts = df["tracking_state"].value_counts().to_dict()
                    for state, count in state_counts.items():
                        sf.write("- **{}**: {} 帧 ({:.2f}%)\n".format(state, count, count/total_frames*100))
                    sf.write("\n")
                    
                sf.write("## 4. 后期针对性优化与改进意见\n\n")
                for sug in suggestions:
                    sf.write(sug + "\n")
                sf.write("\n---\n*本报告由无人机测距评估系统在处理完成后自动计算并生成。*\n")
                
            print("   已自动生成测试分析报告简介: {}".format(summary_path.name))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("   警告: 自动生成报告失败: {}".format(str(e)))


if __name__ == "__main__":
    root = tk.Tk()
    app = UAVDistanceStabilizerGUI(root)
    root.mainloop()
