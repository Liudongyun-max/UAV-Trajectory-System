"""
UAV trajectory prediction desktop GUI.

The GUI intentionally runs inference one frame at a time from a QTimer instead
of pushing frames from a worker thread. This keeps OpenCV, PyTorch, and Qt image
ownership in one place and makes shutdown deterministic: stopping or closing the
window always releases the writer before the app exits.
"""
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
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import InferenceSession


def write_crash_log(exc_type, exc_value, exc_traceback):
    log_path = Path(__file__).with_name("predict_gui_error.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n=== Unhandled GUI exception ===\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)


sys.excepthook = write_crash_log


class VideoDisplay(QLabel):
    def sizeHint(self):
        return QSize(1, 1)

    def minimumSizeHint(self):
        return QSize(1, 1)


class UAVTrajectoryGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV 轨迹预测与可视化检测系统")
        self.resize(1350, 850)
        self.setMinimumSize(1100, 700)

        self.video_path = ""
        self.tracks_path = ""
        self.session = None
        self.paused = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

        self.init_ui()
        self.scan_preinstalled_models()
        self.log("[INFO] 客户端已启动。")
        self.log("[INFO] 当前版本使用同步逐帧预览，关闭窗口时会先完成视频写入归档。")

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a24; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', 'Microsoft YaHei'; font-size: 12px; }
            QFrame#panel_left { background-color: #242432; border-right: 1px solid #323248; }
            QLabel#title_left { font-size: 16px; font-weight: bold; color: #00e5ff; padding-bottom: 10px; border-bottom: 1px solid #323248; }
            QLineEdit, QComboBox {
                background-color: #1e1e2c; border: 1px solid #3c3c56; border-radius: 4px;
                padding: 5px 8px; color: #ffffff;
            }
            QPushButton {
                background-color: #3f51b5; border: none; border-radius: 4px;
                color: white; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #5c6bc0; }
            QPushButton:disabled { background-color: #2c2c3e; color: #66667e; }
            QPushButton#btn_launch {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00b0ff, stop:1 #00e5ff);
                color: #0d47a1; font-size: 14px;
            }
            QProgressBar {
                border: 1px solid #3c3c56; border-radius: 4px; background-color: #1e1e2c;
                text-align: center; color: #ffffff; font-weight: bold;
            }
            QProgressBar::chunk { background-color: #00e5ff; border-radius: 3px; }
            QTextEdit#console {
                background-color: #0f0f15; border: 1px solid #28283a; border-radius: 6px;
                color: #b8bacb; font-family: 'Consolas', 'Courier New', monospace; font-size: 11px;
            }
        """)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        left_panel = QFrame()
        left_panel.setObjectName("panel_left")
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 20, 15, 20)
        left_layout.setSpacing(15)

        title_label = QLabel("UAV 预测控制台")
        title_label.setObjectName("title_left")
        left_layout.addWidget(title_label)

        video_layout = QVBoxLayout()
        video_layout.addWidget(QLabel("1. 导入待检测视频:"))
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

        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("2. 选择模型权重 (.pth):"))
        model_row = QHBoxLayout()
        self.cb_model = QComboBox()
        self.btn_custom_model = QPushButton("自定义权重...")
        self.btn_custom_model.clicked.connect(self.browse_model)
        model_row.addWidget(self.cb_model)
        model_row.addWidget(self.btn_custom_model)
        model_layout.addLayout(model_row)
        left_layout.addLayout(model_layout)

        csv_layout = QVBoxLayout()
        self.chk_csv = QCheckBox("启用测量轨迹 CSV")
        self.chk_csv.setChecked(True)
        self.chk_csv.toggled.connect(self.toggle_csv_inputs)
        csv_layout.addWidget(self.chk_csv)
        csv_row = QHBoxLayout()
        self.txt_csv = QLineEdit()
        self.txt_csv.setReadOnly(True)
        self.txt_csv.setPlaceholderText("请选择 measured_tracks.csv，可选")
        self.btn_browse_csv = QPushButton("浏览...")
        self.btn_browse_csv.clicked.connect(self.browse_csv)
        csv_row.addWidget(self.txt_csv)
        csv_row.addWidget(self.btn_browse_csv)
        csv_layout.addLayout(csv_row)
        left_layout.addLayout(csv_layout)

        left_layout.addWidget(QLabel("4. 运行参数:"))
        self.chk_smooth = QCheckBox("开启 Savgol 历史轨迹平滑")
        self.chk_smooth.setChecked(True)
        self.chk_save_video = QCheckBox("保存预测视频到 outputs/")
        self.chk_save_video.setChecked(True)
        self.chk_show_pred = QCheckBox("展示预测轨迹 (蓝线/蓝点)")
        self.chk_show_pred.setChecked(True)
        self.chk_show_hist = QCheckBox("展示历史轨迹 (红色折线)")
        self.chk_show_hist.setChecked(True)
        self.chk_show_gt = QCheckBox("展示真实轨迹 (绿线/绿点)")
        self.chk_show_gt.setChecked(True)
        left_layout.addWidget(self.chk_smooth)
        left_layout.addWidget(self.chk_save_video)
        left_layout.addWidget(self.chk_show_pred)
        left_layout.addWidget(self.chk_show_hist)
        left_layout.addWidget(self.chk_show_gt)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("计算设备:"))
        self.cb_device = QComboBox()
        self.cb_device.addItems(["auto", "cuda", "cpu"])
        device_row.addWidget(self.cb_device)
        left_layout.addLayout(device_row)

        self.btn_launch = QPushButton("启动实时检测与预测")
        self.btn_launch.setObjectName("btn_launch")
        self.btn_launch.setFixedHeight(40)
        self.btn_launch.clicked.connect(self.start_processing)
        left_layout.addWidget(self.btn_launch)
        left_layout.addStretch()
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 20, 15, 20)
        right_layout.setSpacing(12)

        self.video_container = QFrame()
        self.video_container.setStyleSheet("background-color: #0c0c12; border: 1px solid #28283a; border-radius: 6px;")
        container_layout = QVBoxLayout(self.video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video = VideoDisplay("导入视频并点击启动后，将在此实时预览预测画面")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(1, 1)
        self.lbl_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_video.setStyleSheet("color: #555570; font-size: 14px;")
        container_layout.addWidget(self.lbl_video)
        right_layout.addWidget(self.video_container, 3)

        control_row = QHBoxLayout()
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.pause_processing)
        self.btn_stop = QPushButton("停止并保存")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.finish_session("手动停止"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(22)
        self.lbl_status = QLabel("状态: 空闲")
        self.lbl_status.setFixedWidth(260)
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        control_row.addWidget(self.btn_pause)
        control_row.addWidget(self.btn_stop)
        control_row.addWidget(self.progress_bar, 1)
        control_row.addWidget(self.lbl_status)
        right_layout.addLayout(control_row)

        right_layout.addWidget(QLabel("控制台输出日志:"))
        self.console = QTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        right_layout.addWidget(self.console, 1)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 970])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

    def scan_preinstalled_models(self):
        self.cb_model.clear()
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists():
            for model_path in sorted(models_dir.glob("*.pth")):
                self.cb_model.addItem(model_path.name, str(model_path))
        if self.cb_model.count() == 0:
            self.cb_model.addItem("未找到内置模型，请选择自定义模型", "")

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择待检测视频", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if not file_path:
            return
        self.video_path = file_path
        self.txt_video.setText(file_path)
        self.log(f"[INFO] 已导入视频: {file_path}")
        if self.chk_csv.isChecked():
            auto_csv = Path(file_path).parent / "measured_tracks.csv"
            if auto_csv.exists():
                self.tracks_path = str(auto_csv)
                self.txt_csv.setText(str(auto_csv))
                self.log(f"[INFO] 自动载入同目录 measured_tracks.csv: {auto_csv}")

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "PyTorch Weights (*.pth);;All Files (*)")
        if not file_path:
            return
        index = self.cb_model.findData(file_path)
        if index == -1:
            self.cb_model.addItem(f"[自定义] {Path(file_path).name}", file_path)
            index = self.cb_model.count() - 1
        self.cb_model.setCurrentIndex(index)
        self.log(f"[INFO] 已选择模型: {file_path}")

    def browse_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择测量轨迹 CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        self.tracks_path = file_path
        self.txt_csv.setText(file_path)
        self.log(f"[INFO] 已导入轨迹 CSV: {file_path}")

    def toggle_csv_inputs(self, checked):
        self.txt_csv.setEnabled(checked)
        self.btn_browse_csv.setEnabled(checked)
        if not checked:
            self.tracks_path = ""
            self.txt_csv.clear()

    def start_processing(self):
        if self.session is not None:
            return
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "参数错误", "请先选择需要检测的视频文件。")
            return
        model_path = self.cb_model.currentData()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "参数错误", "请先选择有效的 .pth 模型权重。")
            return

        self.set_controls_enabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("状态: 正在初始化...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.session = InferenceSession(
                video_path=self.video_path,
                model_path=model_path,
                tracks_path=self.tracks_path if self.chk_csv.isChecked() else None,
                smooth=self.chk_smooth.isChecked(),
                device=self.cb_device.currentText(),
                save_video=self.chk_save_video.isChecked(),
                outputs_root=Path(__file__).parent / "outputs",
                show_pred=self.chk_show_pred.isChecked(),
                show_hist=self.chk_show_hist.isChecked(),
                show_gt=self.chk_show_gt.isChecked(),
            )
            self.flush_session_logs()
            interval_ms = max(1, int(1000.0 / max(1.0, self.session.fps)))
            self.timer.start(interval_ms)
            self.lbl_status.setText("状态: 实时预测中")
            self.log(f"[INFO] 实时预览已启动，输出目录: {self.session.output_dir}")
        except Exception as exc:
            self.log(f"[ERROR] 初始化失败: {exc}")
            self.log(traceback.format_exc())
            self.session = None
            self.set_controls_enabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            QMessageBox.critical(self, "启动失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def process_next_frame(self):
        if self.session is None or self.paused:
            return
        try:
            result = self.session.step()
            self.flush_session_logs()
            if result is None:
                self.finish_session("执行完毕", already_closed=True)
                return
            self.show_frame(result["frame"])
            self.progress_bar.setValue(result["progress"])
            ade = result["ade"]
            if ade > 0:
                self.lbl_status.setText(f"Frame: {result['frame_idx']}/{result['total_frames']}  ADE: {ade:.2f}px")
            else:
                self.lbl_status.setText(f"Frame: {result['frame_idx']}/{result['total_frames']}")
        except Exception as exc:
            self.log(f"[ERROR] 实时预测异常: {exc}")
            self.log(traceback.format_exc())
            self.finish_session(f"错误: {exc}")

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
        self.lbl_status.setText("状态: 暂停" if self.paused else "状态: 实时预测中")

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
            self.log(f"[INFO] 输出视频: {output_path}")

    def set_controls_enabled(self, enabled):
        self.btn_browse_video.setEnabled(enabled)
        self.cb_model.setEnabled(enabled)
        self.btn_custom_model.setEnabled(enabled)
        self.chk_csv.setEnabled(enabled)
        self.btn_browse_csv.setEnabled(enabled and self.chk_csv.isChecked())
        self.chk_smooth.setEnabled(enabled)
        self.chk_save_video.setEnabled(enabled)
        self.chk_show_pred.setEnabled(enabled)
        self.chk_show_hist.setEnabled(enabled)
        self.chk_show_gt.setEnabled(enabled)
        self.cb_device.setEnabled(enabled)
        self.btn_launch.setEnabled(enabled)

    def flush_session_logs(self):
        if self.session is None:
            return
        for line in self.session.drain_logs():
            self.log(line)

    def log(self, text):
        self.console.append(str(text))
        self.console.ensureCursorVisible()

    def closeEvent(self, event):
        if self.session is not None:
            self.log("[INFO] 窗口关闭，正在同步保存当前已处理的视频结果...")
            self.finish_session("手动关闭")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = UAVTrajectoryGUI()
    window.show()
    sys.exit(app.exec_())
