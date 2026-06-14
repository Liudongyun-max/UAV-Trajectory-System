"""
模型导出入口脚本

用法:
    python scripts/export_model.py --model models/gru_baseline_final.pth --format onnx
    python scripts/export_model.py --model models/gru_baseline_final.pth --format torchscript
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from training.models.gru_trajectory import UAVTrajectoryNet
from training.exporters.onnx_export import export_to_onnx
from training.exporters.torchscript_export import export_to_torchscript


def main():
    parser = argparse.ArgumentParser(description="Export UAV Trajectory Model")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], required=True)
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=20)
    args = parser.parse_args()
    
    checkpoint = torch.load(args.model, map_location="cpu")
    config = checkpoint.get("model_config", {})
    
    model = UAVTrajectoryNet(
        input_dim=config.get("input_dim", args.input_dim),
        hidden_dim=config.get("hidden_dim", 32),
        num_layers=config.get("num_layers", 1),
        future_steps=config.get("future_steps", 5),
        dropout=config.get("dropout", 0.1),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    if args.output:
        output_path = args.output
    else:
        output_path = f"models/uav_trajectory.{args.format}"
    
    if args.format == "onnx":
        export_to_onnx(
            model,
            save_path=output_path,
            input_dim=config.get("input_dim", args.input_dim),
            seq_len=args.seq_len,
        )
    elif args.format == "torchscript":
        export_to_torchscript(
            model,
            save_path=output_path,
            input_dim=config.get("input_dim", args.input_dim),
            seq_len=args.seq_len,
        )
    
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    main()
