"""负样本生成器"""
import numpy as np
from typing import Dict, List, Optional


class NegativeSampleGenerator:
    """
    负样本生成器
    
    生成背景噪声轨迹用于训练:
    1. 随机游走 - 无净位移
    2. 高频振荡 - 快速往复
    3. 静态点 - 固定位置+小抖动
    4. 鸟类轨迹 - 平滑但不同于无人机
    """
    
    def __init__(
        self,
        seq_len: int = 20,
        future_steps: int = 5,
        input_dim: int = 8,
        image_width: int = 2560,
        image_height: int = 1440,
        seed: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.input_dim = input_dim
        self.image_width = image_width
        self.image_height = image_height
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate(self, n_samples: int) -> List[Dict]:
        """
        生成负样本
        
        Args:
            n_samples: 生成数量
        
        Returns:
            负样本列表
        """
        samples = []
        
        n_per_type = n_samples // 4
        
        for _ in range(n_per_type):
            traj = self._random_walk_trajectory()
            samples.append(self._traj_to_sample(traj, "random_walk"))
        
        for _ in range(n_per_type):
            traj = self._oscillation_trajectory()
            samples.append(self._traj_to_sample(traj, "oscillation"))
        
        for _ in range(n_per_type):
            traj = self._static_trajectory()
            samples.append(self._traj_to_sample(traj, "static"))
        
        for _ in range(n_samples - len(samples)):
            traj = self._bird_trajectory()
            samples.append(self._traj_to_sample(traj, "bird"))
        
        return samples
    
    def _traj_to_sample(self, traj: np.ndarray, noise_type: str) -> Dict:
        """将轨迹转换为训练样本"""
        history = traj[:self.seq_len]
        future = traj[self.seq_len:self.seq_len + self.future_steps]
        
        x_norm = history[:, 0]
        y_norm = history[:, 1]
        
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
            # Simulate box height/width (15 to 50 pixels) and normalized aspect ratio
            w_pixels = np.random.uniform(15.0, 50.0, size=self.seq_len)
            h_pixels = np.random.uniform(15.0, 50.0, size=self.seq_len)
            w_norm = w_pixels / float(self.image_width)
            h_norm = h_pixels / float(self.image_height)
            aspect_ratio = w_norm / (h_norm + 1e-8)
            # Background noise detection confidence (typically lower, e.g., 0.3 to 0.8)
            conf = np.random.uniform(0.3, 0.8, size=self.seq_len)
            
            feature_list.extend([w_norm, h_norm, aspect_ratio, conf])
            
        features = np.stack(feature_list, axis=-1).astype(np.float32)
        
        future_offsets = np.stack([
            future[:, 0] - x_norm[-1],
            future[:, 1] - y_norm[-1],
        ], axis=-1).astype(np.float32)
        
        return {
            "features": features,
            "future_offsets": future_offsets,
            "label": 0.0,
            "episode_id": f"synthetic_{noise_type}",
        }
    
    def _random_walk_trajectory(self) -> np.ndarray:
        """生成随机游走轨迹"""
        n_steps = self.seq_len + self.future_steps
        
        start = np.random.uniform(0.3, 0.7, size=2)
        deltas = np.random.normal(0, 0.002, size=(n_steps, 2))
        traj = np.cumsum(deltas, axis=0) + start
        
        return np.clip(traj, 0.1, 0.9)
    
    def _oscillation_trajectory(self) -> np.ndarray:
        """生成高频振荡轨迹"""
        n_steps = self.seq_len + self.future_steps
        t = np.linspace(0, 2 * np.pi, n_steps)
        
        amplitude = np.random.uniform(0.005, 0.02)
        frequency = np.random.uniform(2.0, 5.0)
        
        center = np.random.uniform(0.3, 0.7, size=2)
        
        x = center[0] + amplitude * np.sin(2 * np.pi * frequency * t)
        y = center[1] + amplitude * np.cos(2 * np.pi * frequency * t + np.random.uniform(0, 2 * np.pi))
        
        traj = np.stack([x, y], axis=-1)
        return np.clip(traj, 0.1, 0.9)
    
    def _static_trajectory(self) -> np.ndarray:
        """生成静态点轨迹"""
        n_steps = self.seq_len + self.future_steps
        
        center = np.random.uniform(0.3, 0.7, size=2)
        noise = np.random.normal(0, 0.001, size=(n_steps, 2))
        
        traj = center + noise
        return np.clip(traj, 0.1, 0.9)
    
    def _bird_trajectory(self) -> np.ndarray:
        """生成鸟类轨迹（平滑但不同于无人机）"""
        n_steps = self.seq_len + self.future_steps
        t = np.linspace(0, 1, n_steps)
        
        start = np.random.uniform(0.2, 0.4, size=2)
        end = np.random.uniform(0.6, 0.8, size=2)
        
        base_x = start[0] + (end[0] - start[0]) * t
        base_y = start[1] + (end[1] - start[1]) * t
        
        wave = np.random.uniform(0.01, 0.03) * np.sin(2 * np.pi * 3 * t)
        
        x = base_x + wave
        y = base_y + wave * 0.5
        
        traj = np.stack([x, y], axis=-1)
        return np.clip(traj, 0.1, 0.9)
