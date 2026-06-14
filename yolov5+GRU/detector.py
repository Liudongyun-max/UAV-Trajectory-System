"""
YOLOv5 四驱目标检测器接口 (detector.py)
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class YoloDetector:
    """
    统一的 YOLOv5 目标检测器基类
    """
    def __init__(self, model_path: Optional[str] = None, conf_thres: float = 0.3, nms_thres: float = 0.45):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        对输入 Frame 进行目标检测
        返回: ndarray 包含检测框, 每一行格式为 [x1, y1, x2, y2, confidence, class_id]
        """
        raise NotImplementedError


class RKNNImpl(YoloDetector):
    """
    瑞芯微 RKNN 推理实现 (物理板端环境)
    """
    def __init__(self, model_path: str, conf_thres: float = 0.3, nms_thres: float = 0.45):
        super().__init__(model_path, conf_thres, nms_thres)
        try:
            from rknnlite.api import RKNNLite
        except ImportError as e:
            raise RuntimeError("RKNNLite is not installed in the current environment.") from e
            
        self.rknn = RKNNLite()
        print(f"--> [YoloDetector] Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Load RKNN failed: {ret}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"Init RKNN runtime failed: {ret}")
        
    def detect(self, frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        outputs = self.rknn.inference(inputs=[frame_resized])
        return np.empty((0, 6), dtype=np.float32)


class ONNXImpl(YoloDetector):
    """
    ONNXRuntime 推理实现 (通用 PC 验证环境)
    """
    def __init__(self, model_path: str, conf_thres: float = 0.3, nms_thres: float = 0.45):
        super().__init__(model_path, conf_thres, nms_thres)
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime is not installed. Please run: pip install onnxruntime") from e
            
        print(f"--> [YoloDetector] Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
    def detect(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        outputs = self.session.run(None, {self.input_name: img})
        pred = outputs[0][0]
        
        obj_conf = pred[:, 4]
        class_conf = np.max(pred[:, 5:], axis=1)
        class_id = np.argmax(pred[:, 5:], axis=1)
        scores = obj_conf * class_conf
        
        keep = scores > self.conf_thres
        if not np.any(keep):
            return np.empty((0, 6), dtype=np.float32)
            
        valid = pred[keep]
        valid_scores = scores[keep]
        valid_class_id = class_id[keep]
        
        boxes = []
        for i, det in enumerate(valid):
            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            x1 = (cx - bw / 2.0) * (w / 640.0)
            y1 = (cy - bh / 2.0) * (h / 640.0)
            bw_orig = bw * (w / 640.0)
            bh_orig = bh * (h / 640.0)
            boxes.append([float(x1), float(y1), float(bw_orig), float(bh_orig)])
            
        indices = cv2.dnn.NMSBoxes(boxes, valid_scores.tolist(), self.conf_thres, self.nms_thres)
        if len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)
            
        res = []
        for idx in np.array(indices).flatten():
            box = boxes[idx]
            x1, y1, bw_o, bh_o = box
            res.append([x1, y1, x1 + bw_o, y1 + bh_o, float(valid_scores[idx]), int(valid_class_id[idx])])
            
        return np.array(res, dtype=np.float32)


class PyTorchImpl(YoloDetector):
    """
    PyTorch 推理实现 (PyTorch 离线测试环境)
    """
    def __init__(self, model_path: str, conf_thres: float = 0.3, nms_thres: float = 0.45, device: str = "cpu"):
        super().__init__(model_path, conf_thres, nms_thres)
        import torch
        print(f"--> [YoloDetector] Loading PyTorch custom model: {model_path} on {device}")
        
        # 1. 尝试使用本地已有的完整 YOLOv5 源码进行完全离线加载 (无需网络，最鲁棒)
        local_yolo_paths = [
            "F:/Obsidian/Smart Brain/09_航拍视觉与YOLO资料/yolov5-master",
            "F:/Obsidian/Smart Brain/09_航拍视觉与YOLO资料/yolov5-7.0",
        ]
        
        loaded = False
        for path in local_yolo_paths:
            if os.path.exists(os.path.join(path, "hubconf.py")):
                try:
                    print(f"--> [YoloDetector] Attempting offline load using local source: {path}")
                    self.model = torch.hub.load(path, 'custom', path=model_path, source='local', device=device)
                    loaded = True
                    print("--> [YoloDetector] Offline load successful!")
                    break
                except Exception as e:
                    print(f"--> [YoloDetector] Offline load from {path} failed: {e}")
                    
        # 2. 如果本地路径均不存在或加载失败，尝试使用默认 hub 联网下载并重置缓存 (force_reload=True)
        if not loaded:
            try:
                print("--> [YoloDetector] Local source load failed or not found. Attempting torch.hub online load (force_reload=True)...")
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, device=device)
                loaded = True
                print("--> [YoloDetector] Online load successful!")
            except Exception as e:
                raise RuntimeError(f"All PyTorch load methods failed: {e}")
                
        self.model.conf = conf_thres
        self.model.iou = nms_thres
        
    def detect(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame)
        pred = results.xyxy[0].cpu().numpy()
        return pred


class MockFallbackImpl(YoloDetector):
    """
    Mock Fallback 推理实现 (零参数物理特征定位器)
    """
    def __init__(self, conf_thres: float = 0.3, nms_thres: float = 0.45):
        super().__init__(None, conf_thres, nms_thres)
        print("--> [YoloDetector] Warning: No model weights provided. Falling back to Adaptive Local Contrast & Spatial Isolation Fallback Detector.")
        
    def detect(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        bg = cv2.blur(gray, (15, 15))
        diff = cv2.absdiff(bg, gray)
        
        _, thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        border = 8
        centroids = []
        
        # 按照面积从大到小限制最多处理前 150 个，防止复杂背景下噪点过多导致 KDTree 等计算开销过大
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3], reverse=True)[:150]
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if 4 <= area < 400:
                cX = int(x + w / 2.0)
                cY = int(y + h / 2.0)
                if border <= cX < width - border and border <= cY < height - border:
                    max_contrast = int(np.max(diff[y:y+h, x:x+w]))
                    centroids.append((cX, cY, area, max_contrast, x, y, w, h))
                    
        if not centroids:
            return np.empty((0, 6), dtype=np.float32)
            
        coords = np.array([[p[0], p[1]] for p in centroids])
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        counts = tree.query_ball_point(coords, r=150, return_length=True)
        neighbors = counts - 1
        
        isolated_dets = []
        for i, p in enumerate(centroids):
            if neighbors[i] < 4:
                score = p[2] * p[3]
                isolated_dets.append((p, score))
                
        if not isolated_dets:
            return np.empty((0, 6), dtype=np.float32)
            
        isolated_dets.sort(key=lambda x: x[1], reverse=True)
        
        res = []
        for item, score_val in isolated_dets:
            cX, cY, area, max_contrast, rx, ry, rw, rh = item
            pad_w = max(15, rw)
            pad_h = max(15, rh)
            x1 = max(0, cX - pad_w)
            y1 = max(0, cY - pad_h)
            x2 = min(width - 1, cX + pad_w)
            y2 = min(height - 1, cY + pad_h)
            confidence = min(0.99, max(0.40, 0.5 + 0.05 * max_contrast))
            res.append([x1, y1, x2, y2, confidence, 0])
            
        return np.array(res, dtype=np.float32)


def get_detector(model_path: Optional[str] = None, conf_thres: float = 0.3, nms_thres: float = 0.45, device: str = "cpu") -> YoloDetector:
    """
    自适应获取最合适的目标检测器实例
    """
    if not model_path or not os.path.exists(model_path):
        return MockFallbackImpl(conf_thres, nms_thres)
        
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".rknn":
        return RKNNImpl(model_path, conf_thres, nms_thres)
    elif ext == ".onnx":
        try:
            return ONNXImpl(model_path, conf_thres, nms_thres)
        except Exception as e:
            print(f"--> [YoloDetector] Load ONNX failed ({e}). Falling back to Mock.")
            return MockFallbackImpl(conf_thres, nms_thres)
    elif ext in (".pt", ".pth"):
        try:
            return PyTorchImpl(model_path, conf_thres, nms_thres, device=device)
        except Exception as e:
            print(f"--> [YoloDetector] Load PyTorch failed ({e}). Falling back to Mock.")
            return MockFallbackImpl(conf_thres, nms_thres)
    else:
        return MockFallbackImpl(conf_thres, nms_thres)
