#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401
from training.data.distance_sequence_builder import DistanceSequenceBuilder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    source_root = Path(cfg["data"]["source_root"])
    seq_len = cfg["data"]["sequence_length"]
    stride = cfg["data"]["stride"]
    scene_features = cfg["data"].get("scene_features", False)
    
    # 构造 sequences
    folds_data = DistanceSequenceBuilder.build_sequences(
        source_root=source_root,
        sequence_length=seq_len,
        stride=stride,
        scene_features=scene_features
    )
    
    out_dir = Path("outputs") / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为 npz 压缩格式
    save_path = out_dir / "distance_sequences.npz"
    save_data = {}
    for f_id in range(5):
        if folds_data[f_id] is not None:
            save_data[f"fold_{f_id}_x"] = folds_data[f_id]["x"]
            save_data[f"fold_{f_id}_y"] = folds_data[f_id]["y"]
            save_data[f"fold_{f_id}_missing"] = folds_data[f_id]["missing_mask"]
            save_data[f"fold_{f_id}_interpolated"] = folds_data[f_id]["interpolated_mask"]
            
            # 元数据转为 json string
            meta_json = json.dumps(folds_data[f_id]["meta"])
            save_data[f"fold_{f_id}_meta"] = np.array([meta_json], dtype=object)
    np.savez_compressed(save_path, **save_data)
    print(f"Compressed sequence dataset saved to {save_path}")
    
    # 输出简短的 split_manifest.csv
    manifest_rows = []
    for f_id in range(5):
        if folds_data[f_id] is not None:
            manifest_rows.append({
                "fold_id": f_id,
                "sequence_count": len(folds_data[f_id]["x"]),
                "effective_fps_mix": "25fps_50%_15fps_50%"
            })
    pd.DataFrame(manifest_rows).to_csv(out_dir / "split_manifest.csv", index=False)
    
    # 拷贝 feature_schema.json 和 normalization.json
    import shutil
    shutil.copy(source_root / "feature_schema.json", out_dir / "feature_schema.json")
    shutil.copy(source_root / "normalization.json", out_dir / "normalization.json")
    
    # 保存生效的最终 yaml 配置
    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
        
    print(f"Sequences building completed successfully.")


if __name__ == "__main__":
    main()
