"""日志配置"""
import logging
from pathlib import Path


def setup_logger(
    log_file: str = "training.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    设置日志
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        Logger实例
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    
    return logging.getLogger(__name__)
