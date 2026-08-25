from loguru import logger
import sys
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

logger.add(
    "logs/airsense.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO",
)