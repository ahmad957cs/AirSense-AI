from loguru import logger
import sys
from pathlib import Path

# Create logs folder
Path("logs").mkdir(exist_ok=True)

# Remove default logger
logger.remove()

# Console logger
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# File logger
logger.add(
    "logs/airsense.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO",
)