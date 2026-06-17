#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DistanceSequenceDataset(Dataset):
    def __init__(self, fold_dict: dict):
        self.x = torch.from_numpy(fold_dict["x"])
        self.y = torch.from_numpy(fold_dict["y"])
        self.missing_mask = torch.from_numpy(fold_dict["missing_mask"])
        self.interpolated_mask = torch.from_numpy(fold_dict["interpolated_mask"])
        self.meta = fold_dict["meta"]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        return (
            self.x[idx],
            self.y[idx],
            self.missing_mask[idx],
            self.interpolated_mask[idx],
            self.meta[idx]
        )
