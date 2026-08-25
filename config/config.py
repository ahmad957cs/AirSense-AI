from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

AQICN_API_KEY = os.getenv("AQICN_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

DEFAULT_CITY = os.getenv("CITY", "Islamabad")

AQICN_BASE_URL = "https://api.waqi.info/feed"

REQUEST_TIMEOUT = 30
