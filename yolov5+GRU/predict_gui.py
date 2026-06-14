import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent))

from gui_inference import GuiInferenceSession

def write_crash_log(exc_type, exc_value, exc_traceback):
    log_path = Path(__file__).with_name("predict_gui_error.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n=== Unhandled GUI exception ===\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = write_crash_log

class VideoDisplay(QLabel):
    def sizeHint(self):
        return QSize(1, 1)

    def minimumSizeHint(self):
        return QSize(1, 1)

class UAVTrajectoryGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 + GRU 级联追踪与未来轨迹预测部署系统")
        self.resize(1400, 880)
        self.setMinimumSize(1150, 720)

        self.video_path = ""
        self.yolo_model_path = ""
        self.gru_model_path = ""
        self.session = None
        self.paused = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

        self.init_ui()
        self.scan_models()
        self.log("[INFO] 级联可视化检测系统客户端已拉起。")
        self.log("[INFO] 支持 YOLOv5 检测与 GRU 时序预测的双驱融合。")

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121218; }
            QWidget { color: #dcdce5; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 12px; }
            QFrame#panel_left { background-color: #1c1c28; border-right: 1px solid #28283a; }
            QLabel#title_left { font-size: 16px; font-weight: bold; color: #00e5ff; padding-bottom: 8px; border-bottom: 1px solid #28283a; }
            QLineEdit, QComboBox {
                background-color: #161622; border: 1px solid #32324a; border-radius: 4px;
                padding: 4px 6px; color: #ffffff;
            }
            QPushButton {
                background-color: #3b50df; border: none; border-radius: 4px;
                color: white; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #5569f5; }
            QPushButton:disabled { background-color: #242436; color: #555570; }
            QPushButton#btn_launch {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00b0ff, stop:1 #00e5ff);
                color: #0b2265; font-size: 14px;
            }
            QProgressBar {
                border: 1px solid #32324a; border-radius: 4px; background-color: #161622;
                text-align: center; color: #ffffff; font-weight: bold;
            }
            QProgressBar::chunk { background-color: #00e5ff; border-radius: 3px; }
            QTextEdit#console {
                background-color: #08080d; border: 1px solid #222232; border-radius: 6px;
                color: #abb2bf; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #28283a; height: 6px; background: #161622; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00e5ff; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px;
            }
        """)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        left_panel = QFrame()
        left_panel.setObjectName("panel_left")
        left_panel.setFixedWidth(390)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 20, 15, 20)
        left_layout.setSpacing(15)

        title_label = QLabel("YOLO+GRU 可视化部署")
        title_label.setObjectName("title_left")
        left_layout.addWidget(title_label)

        # 1. 导入视频
        video_layout = QVBoxLayout()
        video_layout.addWidget(QLabel("1. 导入航拍检测视频:"))
        video_row = QHBoxLayout()
        self.txt_video = QLineEdit()
        self.txt_video.setReadOnly(True)
        self.txt_video.setPlaceholderText("请选择视频文件...")
        self.btn_browse_video = QPushButton("浏览...")
        self.btn_browse_video.clicked.connect(self.browse_video)
        video_row.addWidget(self.txt_video)
        video_row.addWidget(self.btn_browse_video)
        video_layout.addLayout(video_row)
        left_layout.addLayout(video_layout)

        # 2. YOLO 权重
        yolo_layout = QVBoxLayout()
        yolo_layout.addWidget(QLabel("2. 选择 YOLOv5 目标检测权重 (.pt/.onnx/.rknn):"))
        yolo_row = QHBoxLayout()
        self.cb_yolo = QComboBox()
        self.btn_yolo_custom = QPushButton("自定义...")
        self.btn_yolo_custom.clicked.connect(self.browse_yolo)
        yolo_row.addWidget(self.cb_yolo)
        yolo_row.addWidget(self.btn_yolo_custom)
        yolo_layout.addLayout(yolo_row)
        left_layout.addLayout(yolo_layout)

        # 3. GRU 权重
        gru_layout = QVBoxLayout()
        gru_layout.addWidget(QLabel("3. 选择 GRU 时序预测权重 (.pth):"))
        gru_row = QHBoxLayout()
        self.cb_gru = QComboBox()
        self.btn_gru_custom = QPushButton("自定义...")
        self.btn_gru_custom.clicked.connect(self.browse_gru)
        gru_row.addWidget(self.cb_gru)
        gru_row.addWidget(self.btn_gru_custom)
        gru_layout.addLayout(gru_row)
        left_layout.addLayout(gru_layout)

        # 4. 参数配置
        left_layout.addWidget(QLabel("4. 运行参数一键调节:"))
        
        # YOLO 置信度滑块
        conf_layout = QVBoxLayout()
        conf_label_row = QHBoxLayout()
        conf_label_row.addWidget(QLabel("YOLO 检测阈值 (Conf):"))
        self.lbl_conf_val = QLabel("0.40")
        self.lbl_conf_val.setStyleSheet("color: #00e5ff; font-weight: bold;")
        conf_label_row.addWidget(self.lbl_conf_val, 0, Qt.AlignRight)
        conf_layout.addLayout(conf_label_row)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(40) # 默认置信度 0.40
        self.slider_conf.valueChanged.connect(self.on_conf_slider_changed)
        conf_layout.addWidget(self.slider_conf)
        left_layout.addLayout(conf_layout)

        # 个性化显示控制
        self.chk_smooth = QCheckBox("启用 Savitzky-Golay 历史平滑")
        self.chk_smooth.setChecked(True)
        
        self.chk_show_box = QCheckBox("展示 YOLO 黄色定位目标框")
        self.chk_show_box.setChecked(True)
        self.chk_show_box.toggled.connect(self.on_render_flags_changed)
        
        self.chk_show_history = QCheckBox("展示绿色历史航迹轨迹线")
        self.chk_show_history.setChecked(True)
        self.chk_show_history.toggled.connect(self.on_render_flags_changed)
        
        self.chk_show_red_blue = QCheckBox("个性化显示：展示红线与蓝点预测航迹")
        self.chk_show_red_blue.setChecked(False) # 默认 False（显示常规青色）
        self.chk_show_red_blue.toggled.connect(self.on_render_flags_changed)
        
        self.chk_save_video = QCheckBox("保存检测视频到 outputs/")
        self.chk_save_video.setChecked(True)

        left_layout.addWidget(self.chk_smooth)
        left_layout.addWidget(self.chk_show_box)
        left_layout.addWidget(self.chk_show_history)
        left_layout.addWidget(self.chk_show_red_blue)
        left_layout.addWidget(self.chk_save_video)

        # 设备选择
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("物理计算设备:"))
        self.cb_device = QComboBox()
        self.cb_device.addItems(["cpu", "cuda", "auto"])
        device_row.addWidget(self.cb_device)
        left_layout.addLayout(device_row)

        self.btn_launch = QPushButton("启动 YOLO+GRU 级联预测部署")
        self.btn_launch.setObjectName("btn_launch")
        self.btn_launch.setFixedHeight(42)
        self.btn_launch.clicked.connect(self.start_processing)
        left_layout.addWidget(self.btn_launch)
        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # 右面板 (视频预览 + 控制台)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 20, 15, 20)
        right_layout.setSpacing(12)

        self.video_container = QFrame()
        self.video_container.setStyleSheet("background-color: #08080c; border: 1px solid #222232; border-radius: 6px;")
        container_layout = QVBoxLayout(self.video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video = VideoDisplay("视频就绪后点击启动，将在此实时预览 YOLOv5 锁定与 GRU 未来航迹")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(1, 1)
        self.lbl_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_video.setStyleSheet("color: #4b4b60; font-size: 14px;")
        container_layout.addWidget(self.lbl_video)
        right_layout.addWidget(self.video_container, 3)

        # 控制行
        control_row = QHBoxLayout()
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_processing)
        self.btn_stop = QPushButton("停止并归档")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.finish_session("手动停止"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(22)
        self.lbl_status = QLabel("状态: 空闲")
        self.lbl_status.setFixedWidth(280)
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        control_row.addWidget(self.btn_pause)
        control_row.addWidget(self.btn_stop)
        control_row.addWidget(self.progress_bar, 1)
        control_row.addWidget(self.lbl_status)
        right_layout.addLayout(control_row)

        # Console
        right_layout.addWidget(QLabel("控制台输出日志 (系统事件与 YOLO 检测):"))
        self.console = QTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        right_layout.addWidget(self.console, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([390, 1010])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

    def scan_models(self):
        # 1. 扫描 YOLOv5 权重 (F:/UAV Trajectory System/yolov5_model/)
        self.cb_yolo.clear()
        yolo_dir = Path(__file__).resolve().parents[1] / "yolov5_model"
        if yolo_dir.exists():
            for pt in sorted(yolo_dir.glob("*.pt")):
                self.cb_yolo.addItem(pt.name, str(pt))
            for onnx in sorted(yolo_dir.glob("*.onnx")):
                self.cb_yolo.addItem(onnx.name, str(onnx))
            for rknn in sorted(yolo_dir.glob("*.rknn")):
                self.cb_yolo.addItem(rknn.name, str(rknn))
        if self.cb_yolo.count() == 0:
            self.cb_yolo.addItem("无内置权重，请手动选择自定义 YOLO 模型", "")
        else:
            # 默认设为含有 small 的最新权重
            for idx in range(self.cb_yolo.count()):
                if "small" in self.cb_yolo.itemText(idx).lower():
                    self.cb_yolo.setCurrentIndex(idx)
                    break

        # 2. 扫描 GRU 权重
        self.cb_gru.clear()
        candidates = [
            Path(__file__).resolve().parents[1] / "detect" / "models",
            Path(__file__).resolve().parent / "models",
        ]
        scanned = set()
        for folder in candidates:
            if folder.exists():
                for pth in folder.glob("*.pth"):
                    if pth.name not in scanned:
                        self.cb_gru.addItem(pth.name, str(pth))
                        scanned.add(pth.name)
        if self.cb_gru.count() == 0:
            self.cb_gru.addItem("未找到内置 GRU 权重，请浏览...", "")

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择输入视频", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if file_path:
            self.video_path = file_path
            self.txt_video.setText(file_path)
            self.log(f"[INFO] 已导入待测视频: {file_path}")

    def browse_yolo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 模型权重", "", "Weights (*.pt *.onnx *.rknn);;All Files (*)")
        if file_path:
            idx = self.cb_yolo.findData(file_path)
            if idx == -1:
                self.cb_yolo.addItem(f"[自定义] {Path(file_path).name}", file_path)
                idx = self.cb_yolo.count() - 1
            self.cb_yolo.setCurrentIndex(idx)
            self.log(f"[INFO] 自定义 YOLO 权重载入: {file_path}")

    def browse_gru(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 GRU 时序模型权重", "", "GRU Weights (*.pth);;All Files (*)")
        if file_path:
            idx = self.cb_gru.findData(file_path)
            if idx == -1:
                self.cb_gru.addItem(f"[自定义] {Path(file_path).name}", file_path)
                idx = self.cb_gru.count() - 1
            self.cb_gru.setCurrentIndex(idx)
            self.log(f"[INFO] 自定义 GRU 权重载入: {file_path}")

    def on_conf_slider_changed(self, value):
        real_val = value / 100.0
        self.lbl_conf_val.setText(f"{real_val:.2f}")
        if self.session is not None:
            self.session.pipeline.detector.conf_thres = real_val
            # 同时更新 pipeline 中的置信度
            self.session.pipeline.conf_thres = real_val
            self.log(f"[CONFIG] 置信度阈值实时调整为: {real_val:.2f}")

    def on_render_flags_changed(self):
        if self.session is not None:
            self.session.show_box = self.chk_show_box.isChecked()
            self.session.show_history = self.chk_show_history.isChecked()
            self.session.show_pred_red_blue = self.chk_show_red_blue.isChecked()

    def start_processing(self):
        if self.session is not None:
            return
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "启动失败", "请先导入有效的航拍视频。")
            return
        yolo_path = self.cb_yolo.currentData()
        if not yolo_path or not os.path.exists(yolo_path):
            QMessageBox.warning(self, "启动失败", "请选择有效的 YOLOv5 目标检测权重。")
            return
        gru_path = self.cb_gru.currentData()
        if not gru_path or not os.path.exists(gru_path):
            QMessageBox.warning(self, "启动失败", "请选择有效的 GRU 时序模型权重。")
            return

        self.set_controls_enabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("状态: 加载模型中...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            self.session = GuiInferenceSession(
                video_path=self.video_path,
                yolo_model_path=yolo_path,
                gru_model_path=gru_path,
                conf_thres=self.slider_conf.value() / 100.0,
                device=self.cb_device.currentText(),
                save_video=self.chk_save_video.isChecked(),
                show_box=self.chk_show_box.isChecked(),
                show_history=self.chk_show_history.isChecked(),
                show_pred_red_blue=self.chk_show_red_blue.isChecked()
            )
            self.session.pipeline.tracker.smooth = self.chk_smooth.isChecked()
            
            self.flush_session_logs()
            interval_ms = max(1, int(1000.0 / max(1.0, self.session.fps)))
            self.timer.start(interval_ms)
            self.lbl_status.setText("状态: 实时检测中")
            self.log(f"[INFO] 级联推理已启动！")
        except Exception as exc:
            self.log(f"[ERROR] 启动失败: {exc}")
            self.log(traceback.format_exc())
            self.session = None
            self.set_controls_enabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            QMessageBox.critical(self, "系统初始化失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def process_next_frame(self):
        if self.session is None or self.paused:
            return
        try:
            result = self.session.step()
            self.flush_session_logs()
            if result is None:
                self.finish_session("检测完毕", already_closed=True)
                return
                
            self.show_frame(result["frame"])
            self.progress_bar.setValue(result["progress"])
            
            pos = result["pos"]
            pos_str = f"({pos[0]}, {pos[1]})" if pos else "丢失"
            self.lbl_status.setText(f"Frame: {result['frame_idx']}/{result['total_frames']}  Pos: {pos_str}")
        except Exception as exc:
            self.log(f"[ERROR] 检测异常: {exc}")
            self.log(traceback.format_exc())
            self.finish_session(f"错误中止: {exc}")

    def show_frame(self, frame_bgr):
        max_preview_width = 1280
        display_frame = frame_bgr
        if frame_bgr.shape[1] > max_preview_width:
            scale = max_preview_width / float(frame_bgr.shape[1])
            display_frame = cv2.resize(frame_bgr, (max_preview_width, max(1, int(frame_bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_video.setPixmap(pixmap.scaled(self.lbl_video.width(), self.lbl_video.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def pause_processing(self):
        if self.session is None:
            return
        self.paused = not self.paused
        self.btn_pause.setText("继续" if self.paused else "暂停")
        self.lbl_status.setText("状态: 暂停" if self.paused else "状态: 实时检测中")

    def finish_session(self, status, already_closed=False):
        self.timer.stop()
        output_path = ""
        final_status = status
        if self.session is not None:
            if already_closed:
                final_status, output_path = status, self.session.output_path
            else:
                final_status, output_path = self.session.close(status)
            self.flush_session_logs()
            self.session = None

        self.paused = False
        self.btn_pause.setText("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.set_controls_enabled(True)
        self.lbl_status.setText(f"状态: {final_status}")
        if output_path:
            self.log(f"[INFO] 视频归档成功: {output_path}")

    def set_controls_enabled(self, enabled):
        self.btn_browse_video.setEnabled(enabled)
        self.cb_yolo.setEnabled(enabled)
        self.btn_yolo_custom.setEnabled(enabled)
        self.cb_gru.setEnabled(enabled)
        self.btn_gru_custom.setEnabled(enabled)
        self.slider_conf.setEnabled(enabled)
        self.chk_smooth.setEnabled(enabled)
        self.chk_save_video.setEnabled(enabled)
        self.cb_device.setEnabled(enabled)
        self.btn_launch.setEnabled(enabled)

    def flush_session_logs(self):
        if self.session is None:
            return
        for line in self.session.drain_logs():
            self.log(line)

    def closeEvent(self, event):
        if self.session is not None:
            self.log("[INFO] 窗口关闭，正在同步释放视频与模型资源并归档...")
            self.finish_session("窗口关闭")
        event.accept()

    def log(self, text):
        self.console.append(str(text))
        self.console.ensureCursorVisible()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = UAVTrajectoryGUI()
    window.show()
    sys.exit(app.exec_())
