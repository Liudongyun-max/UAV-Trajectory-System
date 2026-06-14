import os
import sys
import io
import cv2
import torch
import numpy as np
from pathlib import Path
from pipeline import YoloGRUPipeline

class StdoutRedirector:
    def __init__(self):
        self.buffer = io.StringIO()
        self.terminal = sys.stdout
        
    def write(self, message):
        self.buffer.write(message)
        self.terminal.write(message)
        
    def flush(self):
        self.buffer.flush()
        self.terminal.flush()
        
    def get_and_clear(self):
        val = self.buffer.getvalue()
        self.buffer.close()
        self.buffer = io.StringIO()
        return val

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

class GuiInferenceSession:
    def __init__(
        self,
        video_path,
        yolo_model_path,
        gru_model_path,
        conf_thres=0.4,
        nms_thres=0.45,
        device="cpu",
        save_video=True,
        outputs_root=None,
        show_box=True,
        show_history=True,
        show_pred_red_blue=False
    ):
        self.video_path = str(video_path)
        self.yolo_model_path = str(yolo_model_path)
        self.gru_model_path = str(gru_model_path)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.device = device
        self.save_video = bool(save_video)
        self.outputs_root = Path(outputs_root) if outputs_root else Path(__file__).resolve().parent / "outputs"
        
        self.show_box = show_box
        self.show_history = show_history
        self.show_pred_red_blue = show_pred_red_blue
        
        self.finished = False
        self.output_path = ""
        self.temp_output_path = ""
        
        self.redirector = StdoutRedirector()
        self.old_stdout = sys.stdout
        sys.stdout = self.redirector
        
        self.logs = []
        self._log("[INFO] 正在初始化 YOLOv5 + GRU 级联推理引擎...")
        
        # 打开视频获取规格
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
            
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(self.fps) or self.fps <= 1e-3:
            self.fps = 20.0
            
        self._log(f"[INFO] 视频规格: {self.width}x{self.height} @ {self.fps:.2f} FPS, 共 {self.total_frames} 帧")
        
        # 实例化 Pipeline
        self.pipeline = YoloGRUPipeline(
            video_path=self.video_path,
            yolo_model_path=self.yolo_model_path,
            gru_model_path=self.gru_model_path,
            device=self.device,
            conf_thres=self.conf_thres,
            nms_thres=self.nms_thres
        )
        
        self.frame_idx = 0
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        
        if self.save_video:
            archive_name = Path(self.video_path).stem
            self.output_path = str(self.outputs_root / f"{archive_name}_predicted_cascade.avi")
            self.temp_output_path = str(self.outputs_root / f".{archive_name}_predicted_cascade.writing.avi")
            if os.path.exists(self.temp_output_path):
                os.remove(self.temp_output_path)
            # 使用 XVID 格式或 MJPG
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(self.temp_output_path, fourcc, self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                self.writer.release()
                raise RuntimeError(f"无法创建输出视频文件: {self.temp_output_path}")
            self._log(f"[INFO] 已开启实时视频保存，正在写入: {self.temp_output_path}")
        else:
            self.writer = None

    def _log(self, text):
        self.logs.append(text)

    def drain_logs(self):
        # 捕获 std 重定向
        raw = self.redirector.get_and_clear()
        lines = [line.strip() for line in raw.split('\n') if line.strip()]
        drained = self.logs + lines
        self.logs = []
        return drained

    def step(self):
        if self.finished:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self.close("执行完毕")
            return None
            
        rendered, drone_pos, is_valid, classification_prob = self.pipeline.process_single_frame(
            frame,
            self.frame_idx,
            show_box=self.show_box,
            show_history=self.show_history,
            show_pred_red_blue=self.show_pred_red_blue
        )
        
        if self.writer is not None:
            self.writer.write(rendered)
            
        result = {
            "frame": rendered,
            "frame_idx": self.frame_idx,
            "total_frames": self.total_frames,
            "progress": int((self.frame_idx + 1) / max(1, self.total_frames) * 100),
            "pos": drone_pos,
            "is_valid": is_valid,
            "prob": classification_prob
        }
        
        self.frame_idx += 1
        return result

    def close(self, status="已停止"):
        if self.finished:
            return status, self.output_path
        self.finished = True
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # 还原 stdout
        sys.stdout = self.old_stdout
        
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
                
        return final_status, self.output_path
