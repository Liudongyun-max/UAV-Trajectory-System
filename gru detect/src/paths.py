import re
from pathlib import Path


GENERIC_VIDEO_NAMES = {"rgb", "video", "input", "source"}


def safe_name(name):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip(" .")
    return cleaned or "video"


def video_archive_name(video_path):
    path = Path(video_path)
    stem = safe_name(path.stem)
    if stem.lower() in GENERIC_VIDEO_NAMES and path.parent.name:
        return safe_name(path.parent.name)
    return stem
