import os
import sys
import subprocess
import imageio_ffmpeg

def convert_videos(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
        
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取 ffmpeg 可执行文件路径
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Using ffmpeg from: {ffmpeg_exe}")
    except Exception as e:
        print(f"Error finding ffmpeg: {e}")
        return

    # 支持的视频后缀
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.wmv', '.m4v')
    
    # 列出所有待转换的文件
    files = os.listdir(input_dir)
    video_files = []
    for f in files:
        f_path = os.path.join(input_dir, f)
        if os.path.isfile(f_path) and f.lower().endswith(video_extensions):
            # 排除已经是 _converted.mp4 的文件，避免循环转换
            if f.lower().endswith('_converted.mp4'):
                continue
            video_files.append(f)
            
    if not video_files:
        print(f"No video files found in '{input_dir}' to convert.")
        return
        
    print(f"Found {len(video_files)} video file(s) to convert:")
    for f in video_files:
        print(f"  - {f}")
        
    for f in video_files:
        input_path = os.path.join(input_dir, f)
        base_name, ext = os.path.splitext(f)
        
        # 决定输出文件名
        if ext.lower() == '.mp4':
            output_name = f"{base_name}_converted.mp4"
        else:
            output_name = f"{base_name}.mp4"
            
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\nConverting: '{f}' -> '{output_name}'...")
        
        # 构建 ffmpeg 转换命令
        # -y: 覆盖输出文件
        # -c:v libx264: 使用 H.264 视频编码
        # -pix_fmt yuv420p: 使用最兼容的像素格式
        # -c:a aac: 使用 AAC 音频编码
        cmd = [
            ffmpeg_exe,
            '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            output_path
        ]
        
        try:
            # 运行命令，允许将输出重定向到控制台
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            # 实时读取输出并打印最后几行或进度
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    stripped = line.strip()
                    if stripped:
                        if stripped.startswith(('frame=', 'size=', 'time=')) or 'fps=' in stripped:
                            print(f"\rProgress: {stripped}", end='', flush=True)
                        else:
                            print(stripped)
            print() # 换行
            
            rc = process.poll()
            if rc == 0:
                print(f"Successfully converted and saved to: {output_path}")
            else:
                print(f"Failed to convert '{f}', ffmpeg returned exit code {rc}")
                
        except Exception as e:
            print(f"An error occurred during conversion of '{f}': {e}")

if __name__ == '__main__':
    # 默认路径为用户提供的路径
    default_dir = r"C:\Users\FUN-C\OneDrive\视频\Captures"
    target_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    convert_videos(target_dir)
