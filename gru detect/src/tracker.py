"""
独立特征追踪与对齐处理器 (tracker.py)
"""
import numpy as np
import cv2
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from typing import Tuple, List, Optional, Dict


class FeatureTracker:
    """
    智能特征追踪与对齐处理器 (双驱模式)
    """
    
    def __init__(
        self,
        seq_len: int = 20,
        input_dim: int = 8,
        smooth: bool = True,
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.smooth = smooth
        self.normalize = normalize
        
        # 内部历史缓存列表，保存Dict元素：{'x', 'y', 'px', 'py', 'w', 'h', 'conf'}
        self.history_buffer: List[Dict] = []
        
        # 系统平移对齐偏置量
        self.last_valid_offset = (0.0, 0.0)
        self.offset_initialized = False
        
        # 连续距离跳跃超限帧数计数器 (用于纯视觉防死锁机制)
        self.invalid_dist_frames = 0
        # 记录最近锁定的目标坐标序列 (用于静止死锁检测)
        self.locked_positions: List[Tuple[int, int]] = []
        
        # 偏置突变允许的最大跳跃门限 (像素)，用于过滤森林边缘等背景树木杂点污染
        self.max_offset_jump = 40.0
        # 纯视频模式下，单帧目标中心最大运动位移门限
        self.max_drone_move = 60.0
        
    def detect_drone_centroid(self, frame: np.ndarray, last_pos: Optional[Tuple[int, int]] = None, thresh_val: int = 120) -> Optional[Tuple[int, int]]:
        """
        基于绝对灰度值反向二值化的检测层 (低于 thresh_val 设为白色前景)
        """
        height, width = frame.shape[:2]
        self.height, self.width = height, width
        
        # 动态计算分辨率相关的缩放因子 (以仿真基准 1280x720 为参考)
        scale_area = (width * height) / (1280.0 * 720.0)
        scale_linear = np.sqrt(scale_area)
        
        # 过滤与定位限制
        sky_height_limit = int(height * (1100.0 / 1440.0))
        r_kdtree = 150.0 * scale_linear
        current_max_drone_move = self.max_drone_move * scale_linear
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 绝对灰度反向二值化处理 (低于 thresh_val 判定为白色前景)
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        border = 40
        centroids = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            # 筛选依据：边界框面积 w * h 在 1 到 2000 像素之间
            if 1 <= area < 2000:
                cX = int(x + w / 2.0)
                cY = int(y + h / 2.0)
                # 排除四周最外围 40px 的相机暗角与字符 UI 噪点
                if border <= cX < width - border and border <= cY < height - border:
                    # 分值以绝对暗度评估：120 - 局部最暗点
                    max_contrast = int(120 - np.min(gray[y:y+h, x:x+w]))
                    centroids.append((cX, cY, area, max_contrast))
                    
        if not centroids:
            return None
            
        # 3. 多模式决策与追踪
        if last_pos is not None:
            # 追踪模式：直接在所有候选重心点 (centroids) 中寻找局域最近点 (不限孤立点)
            best_candidate = None
            min_dist = float('inf')
            for p in centroids:
                dist = np.hypot(p[0] - last_pos[0], p[1] - last_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_candidate = p
            if min_dist < current_max_drone_move:
                return (best_candidate[0], best_candidate[1])
            return None
        else:
            # 检测/全局跳转模式：需进行空间邻域密度过滤 (KDTree 过滤地面群聚大楼/树梢噪点)
            coords = np.array([[p[0], p[1]] for p in centroids])
            tree = KDTree(coords)
            counts = tree.query_ball_point(coords, r=r_kdtree, return_length=True)
            neighbors = counts - 1 # 排除自身
            
            isolated_candidates = []
            for i, p in enumerate(centroids):
                if neighbors[i] < 4:
                    isolated_candidates.append(p)
                    
            if not isolated_candidates:
                return None
                
            # 优先在天空区寻找最佳孤立点，天空候选点为空时直接返回 None，杜绝地面错误跳转
            sky_candidates = [p for p in isolated_candidates if p[1] < sky_height_limit]
            if sky_candidates:
                best = max(sky_candidates, key=lambda x: x[2] * x[3])
                return (best[0], best[1])
            else:
                return None

    def update(self, frame: np.ndarray, csv_row: Optional[Dict] = None) -> Tuple[Optional[Tuple[int, int]], Tuple[float, float], bool]:
        """
        根据当前帧更新滑动窗口：
        - 若 csv_row 存在：数据驱动模式 (含闭环偏置修正)
        - 若 csv_row 为空：纯视频驱动模式 (基于图像分割追踪)
        
        返回:
            drone_pos: 无人机在当前帧的实际显示中心像素 (x, y)，丢失时为 None
            offset: 坐标系的校正偏移量
            is_valid: 滑动历史是否填满 seq_len 帧以进行网络预测
        """
        height, width = frame.shape[:2]
        self.height, self.width = height, width
        
        # 动态计算分辨率相关的缩放因子 (以仿真基准 1280x720 为参考)
        scale_area = (width * height) / (1280.0 * 720.0)
        scale_linear = np.sqrt(scale_area)
        current_max_offset_jump = self.max_offset_jump * scale_linear
        current_max_drone_move = self.max_drone_move * scale_linear
        
        if csv_row is not None:
            # 1. 数据驱动模式 (CSV-Driven)
            current_x = csv_row["x_center"]
            current_y = csv_row["y_center"]
            w = csv_row.get("bbox_width", 50.0)
            h = csv_row.get("bbox_height", 50.0)
            conf = csv_row.get("detector_confidence", 1.0)
            
            # 局域搜寻偏置修正量
            drone_center = self.detect_drone_centroid(frame, (int(current_x), int(current_y)))
            
            # 使用偏置跳跃门限过滤背景噪点，确保偏置外参随相机平滑移动
            if drone_center is not None:
                proposed_offset = (float(drone_center[0] - current_x), float(drone_center[1] - current_y))
                if not self.offset_initialized:
                    self.last_valid_offset = proposed_offset
                    self.offset_initialized = True
                    drone_pos = drone_center
                else:
                    dx = proposed_offset[0] - self.last_valid_offset[0]
                    dy = proposed_offset[1] - self.last_valid_offset[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < current_max_offset_jump:
                        self.last_valid_offset = proposed_offset
                        drone_pos = drone_center
                    else:
                        # 偏置突变，判定为检测到了森林树木杂点，抛弃它并使用上一帧有效偏置
                        drone_pos = (int(current_x + self.last_valid_offset[0]), int(current_y + self.last_valid_offset[1]))
            else:
                if self.offset_initialized:
                    drone_pos = (int(current_x + self.last_valid_offset[0]), int(current_y + self.last_valid_offset[1]))
                else:
                    drone_pos = (int(current_x), int(current_y))
                    
            # 缓存历史：除供预测用的原始 CSV 坐标外，同步存入当前已对齐的物理像素 px, py 用于连线
            self.history_buffer.append({
                'x': current_x, 'y': current_y,
                'px': float(drone_pos[0]), 'py': float(drone_pos[1]),
                'w': w, 'h': h, 'conf': conf
            })
        else:
            # 2. 纯视频驱动模式 (Video-Only)
            last_pos = None
            if len(self.history_buffer) > 0:
                last_pos = (int(self.history_buffer[-1]['px']), int(self.history_buffer[-1]['py']))
                
            # 尝试以常规绝对阈值 120 局域/全局搜寻
            drone_center = self.detect_drone_centroid(frame, last_pos, thresh_val=120)
            
            # 若局域追踪无果，尝试以放宽的绝对阈值 130 在局域范围内搜寻 (仅局域，以防全局大面积噪点)
            if drone_center is None and last_pos is not None:
                drone_center = self.detect_drone_centroid(frame, last_pos, thresh_val=130)
                
            # 若全局未找到，且天空为空，我们也允许再次尝试用放宽阈值 130 全局搜索天空中微弱的目标
            if drone_center is None and last_pos is None:
                drone_center = self.detect_drone_centroid(frame, last_pos=None, thresh_val=130)
                
            if drone_center is not None:
                self.last_valid_offset = (0.0, 0.0)
                
                # 连续帧位移门限过滤，防止偶发噪点引起框闪跳
                if len(self.history_buffer) > 0:
                    last_pt = self.history_buffer[-1]
                    dx = drone_center[0] - last_pt['px']
                    dy = drone_center[1] - last_pt['py']
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < current_max_drone_move:
                        drone_pos = drone_center
                        self.invalid_dist_frames = 0
                    else:
                        # 超过单帧最大位移，判定为噪点，维持上一帧位置并增加无效帧计数
                        self.invalid_dist_frames += 1
                        drone_pos = (int(last_pt['px']), int(last_pt['py']))
                else:
                    drone_pos = drone_center
            else:
                # 局域内或全局均未检测到目标，维持上一帧位置（如有）并增加无效计数
                if len(self.history_buffer) > 0:
                    last_pt = self.history_buffer[-1]
                    drone_pos = (int(last_pt['px']), int(last_pt['py']))
                    self.invalid_dist_frames += 1
                else:
                    drone_pos = None
                    
            # 统一静态死锁监测与自动释放
            if drone_pos is not None:
                self.locked_positions.append(drone_pos)
                if len(self.locked_positions) > 50:
                    self.locked_positions.pop(0)
                    
                is_deadlocked = False
                if len(self.locked_positions) >= 30:
                    recent_30 = self.locked_positions[-30:]
                    xs = [p[0] for p in recent_30]
                    ys = [p[1] for p in recent_30]
                    span_x = max(xs) - min(xs)
                    span_y = max(ys) - min(ys)
                    
                    sky_height_limit = int(height * (1100.0 / 1440.0))
                    is_near_ground = drone_pos[1] >= sky_height_limit
                    
                    # 地面死锁分支：地面区域静止不动超 30 帧 (极差小于 5 * scale_linear)
                    if is_near_ground and span_x < 5.0 * scale_linear and span_y < 5.0 * scale_linear:
                        is_deadlocked = True
                        
                    # 全局静止分支：任意区域（包括天空）完全静止不动超 50 帧 (极差小于 3 * scale_linear)
                    # 且天空中发现了另一个明显的、有位移的新候选点（不是自身）
                    elif len(self.locked_positions) >= 50:
                        recent_50 = self.locked_positions[-50:]
                        xs_50 = [p[0] for p in recent_50]
                        ys_50 = [p[1] for p in recent_50]
                        span_x_50 = max(xs_50) - min(xs_50)
                        span_y_50 = max(ys_50) - min(ys_50)
                        if span_x_50 < 3.0 * scale_linear and span_y_50 < 3.0 * scale_linear:
                            # 首先在天空中用阈值 120 寻找新目标
                            new_global_center = self.detect_drone_centroid(frame, last_pos=None, thresh_val=120)
                            if new_global_center is None:
                                # 没找到则尝试放宽绝对阈值 130
                                new_global_center = self.detect_drone_centroid(frame, last_pos=None, thresh_val=130)
                                
                            if new_global_center is not None:
                                dist_to_new = np.hypot(new_global_center[0] - drone_pos[0], new_global_center[1] - drone_pos[1])
                                if dist_to_new > 10.0 * scale_linear:
                                    is_deadlocked = True
                                
                if is_deadlocked:
                    # 判定死锁，清空缓存，并重新捕获
                    self.history_buffer.clear()
                    self.locked_positions.clear()
                    self.invalid_dist_frames = 0
                    
                    # 尝试重新全局搜寻，优先使用绝对阈值 120，找不到用放宽阈值 130
                    global_center = self.detect_drone_centroid(frame, last_pos=None, thresh_val=120)
                    if global_center is None:
                        global_center = self.detect_drone_centroid(frame, last_pos=None, thresh_val=130)
                        
                    if global_center is not None:
                        drone_pos = global_center
                        self.locked_positions.append(drone_pos)
                        self.history_buffer.append({
                            'x': float(drone_pos[0]), 'y': float(drone_pos[1]),
                            'px': float(drone_pos[0]), 'py': float(drone_pos[1]),
                            'w': 50.0, 'h': 50.0, 'conf': 1.0
                        })
                    else:
                        drone_pos = None
                else:
                    self.history_buffer.append({
                        'x': float(drone_pos[0]), 'y': float(drone_pos[1]),
                        'px': float(drone_pos[0]), 'py': float(drone_pos[1]),
                        'w': 50.0, 'h': 50.0, 'conf': 1.0
                    })
            else:
                self.history_buffer.clear()
                self.locked_positions.clear()
                self.invalid_dist_frames = 0

        # 限制缓存窗口大小
        if len(self.history_buffer) > self.seq_len:
            self.history_buffer.pop(0)
            
        is_valid = len(self.history_buffer) == self.seq_len
        return drone_pos, self.last_valid_offset, is_valid

    def get_features(self) -> np.ndarray:
        """
        提取用于 GRU 预测的时序归一化特征
        """
        x = np.array([pt['x'] for pt in self.history_buffer], dtype=np.float64)
        y = np.array([pt['y'] for pt in self.history_buffer], dtype=np.float64)
        w = np.array([pt['w'] for pt in self.history_buffer], dtype=np.float64)
        h = np.array([pt['h'] for pt in self.history_buffer], dtype=np.float64)
        conf = np.array([pt['conf'] for pt in self.history_buffer], dtype=np.float64)
        
        # 差分前全局平滑
        if self.smooth and len(x) >= 5:
            x = savgol_filter(x, 5, 2)
            y = savgol_filter(y, 5, 2)
            
        if self.normalize:
            # 动态获取当前视频宽度 and 高度，并映射回 1280x720 基准尺度后再除以 2560/1440 进行归一化
            # x_norm = (x * (1280 / w_ref)) / 2560 = x / (2.0 * w_ref)
            w_ref = getattr(self, "width", 1280.0)
            h_ref = getattr(self, "height", 720.0)
            x_norm = x / (2.0 * w_ref)
            y_norm = y / (2.0 * h_ref)
            w_norm = w / (2.0 * w_ref)
            h_norm = h / (2.0 * h_ref)
        else:
            x_norm = x
            y_norm = y
            w_norm = w
            h_norm = h
            
        relative_x = x_norm - x_norm[0]
        relative_y = y_norm - y_norm[0]
        
        velocity_x = np.diff(x_norm, prepend=x_norm[0])
        velocity_y = np.diff(y_norm, prepend=y_norm[0])
        
        acceleration_x = np.diff(velocity_x, prepend=velocity_x[0])
        acceleration_y = np.diff(velocity_y, prepend=velocity_y[0])
        
        feature_list = [
            x_norm, y_norm, relative_x, relative_y,
            velocity_x, velocity_y, acceleration_x, acceleration_y
        ]
        
        if self.input_dim == 12:
            aspect_ratio = w_norm / (h_norm + 1e-8)
            feature_list.extend([w_norm, h_norm, aspect_ratio, conf])
            
        features = np.stack(feature_list, axis=-1)
        return features.astype(np.float32)
