"""数据集划分"""
import os
import random
from typing import List, Tuple, Optional
from pathlib import Path


class EpisodeSplitter:
    """
    Episode数据集划分器
    
    支持按episode_id划分，避免数据泄漏
    """
    
    def __init__(
        self,
        root_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        split_by: str = "episode_id",
        seed: int = 42,
    ):
        """
        初始化划分器
        
        Args:
            root_dir: 数据根目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            split_by: 划分依据 (episode_id)
            seed: 随机种子
        """
        self.root_dir = Path(root_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_by = split_by
        self.seed = seed
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "比例之和必须为1.0"
    
    def discover_episodes(self) -> List[str]:
        """发现所有Episode ID"""
        episodes = []
        for p in self.root_dir.rglob("ideal_tracks.csv"):
            ep_dir = p.parent
            episodes.append(ep_dir.name)
        
        episodes = sorted(list(set(episodes)))
        
        # 尝试读取对应的 batch_summary.json 过滤只保留 B 级样本
        str_path = str(self.root_dir).replace("\\", "/")
        if "export/input" in str_path:
            analysis_dir = Path(str_path.replace("export/input", "export/outputs/analysis"))
            summary_path = analysis_dir / "batch_summary.json"
            if summary_path.exists():
                try:
                    import json
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                    b_grade_eps = {item["episode_id"] for item in summary_data if item.get("grade") == "B"}
                    filtered_episodes = [ep for ep in episodes if ep in b_grade_eps]
                    print(f"Filtering: {len(episodes)} episodes found. Kept {len(filtered_episodes)} B-grade episodes from {summary_path}")
                    episodes = filtered_episodes
                except Exception as e:
                    print(f"Warning: Failed to parse batch_summary.json at {summary_path}: {e}")
            else:
                print(f"Info: batch_summary.json not found at {summary_path}. Using all discovered episodes.")
        
        return sorted(episodes)
    
    def split(
        self, 
        episode_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        划分数据集
        
        Args:
            episode_ids: Episode ID列表，None则自动发现
        
        Returns:
            train_ids, val_ids, test_ids
        """
        if episode_ids is None:
            episode_ids = self.discover_episodes()
        
        random.seed(self.seed)
        shuffled = episode_ids.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train:n_train + n_val]
        test_ids = shuffled[n_train + n_val:]
        
        print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        
        return train_ids, val_ids, test_ids
    
    def split_by_maneuver(self) -> Tuple[List[str], List[str], List[str]]:
        """按机动类型划分（确保每种机动在各集合中都有）"""
        episodes = self.discover_episodes()
        
        maneuver_episodes = {}
        for ep in episodes:
            parts = ep.split("_")
            if len(parts) >= 2:
                maneuver = parts[1]
            else:
                maneuver = "unknown"
            
            if maneuver not in maneuver_episodes:
                maneuver_episodes[maneuver] = []
            maneuver_episodes[maneuver].append(ep)
        
        train_ids, val_ids, test_ids = [], [], []
        
        for maneuver, eps in maneuver_episodes.items():
            random.seed(self.seed)
            shuffled = eps.copy()
            random.shuffle(shuffled)
            
            n = len(shuffled)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            train_ids.extend(shuffled[:n_train])
            val_ids.extend(shuffled[n_train:n_train + n_val])
            test_ids.extend(shuffled[n_train + n_val:])
        
        print(f"Dataset split by maneuver: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        
        return train_ids, val_ids, test_ids
