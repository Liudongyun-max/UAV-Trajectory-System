"""
一键执行脚本 (run.py)
"""
import argparse
from pipeline import YoloGRUPipeline

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 + GRU 时空级联目标检测与轨迹预测系统")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--yolo_model", type=str, default=None, help="YOLOv5模型权重路径 (.rknn, .onnx, .pt)")
    parser.add_argument("--gru_model", type=str, default="F:/UAV Trajectory System/gru detect/models/gru_baseline.pth", help="GRU时序预测模型权重路径")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径 (默认为同级 outputs/ 目录下)")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO置信度门限")
    parser.add_argument("--nms", type=float, default=0.45, help="YOLONMS门限")
    parser.add_argument("--device", type=str, default="auto", help="计算设备 (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # 规范化默认输出路径为当前脚本同级 outputs/ 目录下
    if args.output is None:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(current_dir, "outputs", "output.avi")
        
    pipeline = YoloGRUPipeline(
        video_path=args.video,
        yolo_model_path=args.yolo_model,
        gru_model_path=args.gru_model,
        device=args.device,
        conf_thres=args.conf,
        nms_thres=args.nms
    )
    
    pipeline.run(args.output)

if __name__ == "__main__":
    main()
