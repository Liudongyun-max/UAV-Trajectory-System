"""无人机轨迹数据集"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .negative_generator import NegativeSampleGenerator
from .transforms import FeatureTransform


class UAVTrajectoryDataset(Dataset):
    """
    无人机轨迹数据集
    
    数据流:
        export/input/run001/{maneuver}/{episode}/
            ├── ideal_tracks.csv      -> 标签 (GT)
            ├── measured_tracks.csv   -> 输入 (含噪检测)
            └── truth.csv             -> 3D世界状态 (可选)
    
    生成:
        history_features: [20, 8]  # 20帧历史，8维特征
        future_offsets: [5, 2]     # 未来5帧偏移
        label: 0/1                 # 0=噪声, 1=真实无人机
    """
    
    def __init__(
        self,
        root_dir: str,
        episode_ids: List[str],
        seq_len: int = 20,
        future_steps: int = 5,
        input_dim: int = 8,
        negative_ratio: float = 0.3,
        normalize: bool = True,
        transform: Optional[FeatureTransform] = None,
    ):
        """
        初始化数据集
        
        Args:
            root_dir: 数据根目录
            episode_ids: Episode ID列表
            seq_len: 历史序列长度
            future_steps: 未来预测步数
            negative_ratio: 负样本比例
            normalize: 是否归一化
            transform: 数据变换
        """
        self.root_dir = Path(root_dir)
        self.episode_ids = episode_ids
        self.seq_len = seq_len
        self.future_steps = future_steps
        self.input_dim = input_dim
        self.negative_ratio = negative_ratio
        self.normalize = normalize
        self.transform = transform
        self.negative_generator = NegativeSampleGenerator(
            seq_len=seq_len,
            future_steps=future_steps,
            input_dim=input_dim,
        )
        
        # 预先扫描所有含有 ideal_tracks.csv 的子目录并构建映射，避免重复的磁盘遍历并自动适配各种目录深度
        self.episode_dict = {}
        for p in self.root_dir.rglob("ideal_tracks.csv"):
            self.episode_dict[p.parent.name] = p.parent
            
        self.samples = self._load_all_samples()
    
    def _find_episode_dir(self, episode_id: str) -> Optional[Path]:
        """查找Episode目录"""
        return self.episode_dict.get(episode_id)
    
    def _load_all_samples(self) -> List[Dict]:
        """加载所有Episode并生成训练样本"""
        samples = []
        loaded_count = 0
        for ep_id in self.episode_ids:
            ep_dir = self._find_episode_dir(ep_id)
            if ep_dir is None:
                continue
            
            try:
                df_ideal = pd.read_csv(ep_dir / "ideal_tracks.csv")
                df_measured = pd.read_csv(ep_dir / "measured_tracks.csv")
            except Exception as e:
                print(f"Warning: Failed to load {ep_dir}: {e}")
                continue
            
            # 首尾截断以保护时间轴：
            # 找到第一个可见帧和最后一个可见帧的索引，截取其间所有连续帧，保留中间的遮挡或漏检帧
            visible_indices = df_ideal[df_ideal["visible"] == 1].index
            if len(visible_indices) == 0:
                continue
            start_idx = visible_indices[0]
            end_idx = visible_indices[-1]
            
            df_ideal_vis = df_ideal.iloc[start_idx:end_idx+1].reset_index(drop=True)
            df_measured_vis = df_measured.iloc[start_idx:end_idx+1].reset_index(drop=True)
            
            # 前向/后向填充以平滑处理检测缺失值，确保在测量坐标上没有 NaN 导致的计算错误
            df_measured_vis = df_measured_vis.ffill().bfill()
            
            # 差分前平滑机制 (Episode 级别全局平滑，避免滑窗重叠计算，性能提升300倍)：
            if self.transform and getattr(self.transform, "smooth", False):
                from scipy.signal import savgol_filter
                window = getattr(self.transform, "smooth_window", 5)
                polyorder = getattr(self.transform, "smooth_polyorder", 2)
                if len(df_measured_vis) >= window:
                    df_measured_vis["x_center"] = savgol_filter(df_measured_vis["x_center"].values.astype(np.float64), window, polyorder)
                    df_measured_vis["y_center"] = savgol_filter(df_measured_vis["y_center"].values.astype(np.float64), window, polyorder)
            
            if len(df_ideal_vis) < self.seq_len + self.future_steps:
                continue
            
            for i in range(len(df_ideal_vis) - self.seq_len - self.future_steps):
                history = df_measured_vis.iloc[i:i+self.seq_len]
                future = df_ideal_vis.iloc[i+self.seq_len:i+self.seq_len+self.future_steps]
                
                features = self._compute_features(history)
                future_offsets = self._compute_offsets(future, history.iloc[-1])
                
                samples.append({
                    "features": features,
                    "future_offsets": future_offsets,
                    "label": 1.0,
                    "episode_id": ep_id,
                })
            
            n_neg = int(len(df_ideal_vis) * self.negative_ratio)
            neg_samples = self.negative_generator.generate(n_neg)
            samples.extend(neg_samples)
            
            loaded_count += 1
            if loaded_count % 20 == 0:
                print(f"Loaded {loaded_count}/{len(self.episode_ids)} episodes...", flush=True)
        
        return samples
    
    def _compute_features(self, history: pd.DataFrame) -> np.ndarray:
        """
        计算特征
        
        若 input_dim == 8:
        [x_norm, y_norm, relative_x, relative_y, 
         velocity_x, velocity_y, acceleration_x, acceleration_y]
         
        若 input_dim == 12:
        新增:
        [w_norm, h_norm, aspect_ratio, detector_confidence]
        """
        x = history["x_center"].values.astype(np.float64)
        y = history["y_center"].values.astype(np.float64)
        
        if self.normalize:
            x_norm = x / 2560.0
            y_norm = y / 1440.0
        else:
            x_norm = x
            y_norm = y
        
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
            w = history["bbox_width"].values.astype(np.float64)
            h = history["bbox_height"].values.astype(np.float64)
            if self.normalize:
                w_norm = w / 2560.0
                h_norm = h / 1440.0
            else:
                w_norm = w
                h_norm = h
            aspect_ratio = w_norm / (h_norm + 1e-8)
            conf = history["detector_confidence"].values.astype(np.float64) if "detector_confidence" in history.columns else np.ones_like(x)
            
            feature_list.extend([w_norm, h_norm, aspect_ratio, conf])
            
        features = np.stack(feature_list, axis=-1)
        
        return features.astype(np.float32)
    
    def _compute_offsets(
        self, 
        future: pd.DataFrame, 
        last_history: pd.Series
    ) -> np.ndarray:
        """计算未来5帧相对最后一帧的偏移"""
        future_x = future["x_center"].values / 2560.0
        future_y = future["y_center"].values / 1440.0
        
        last_x = last_history["x_center"] / 2560.0
        last_y = last_history["y_center"] / 1440.0
        
        offsets = np.stack([future_x - last_x, future_y - last_y], axis=-1)
        return offsets.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        features = torch.tensor(sample["features"], dtype=torch.float32)
        future_offsets = torch.tensor(sample["future_offsets"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.float32)
        
        if self.transform:
            features = self.transform(features)
        
        return {
            "history_features": features,
            "future_offsets": future_offsets,
            "label": label,
        }
