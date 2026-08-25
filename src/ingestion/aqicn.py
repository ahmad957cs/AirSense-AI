"""
AQICN API Client
----------------
Fetches real-time Air Quality data from AQICN.

Author: Ahmad
Project: AirSense-AI
"""

import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from config.config import (
    AQICN_API_KEY,
    AQICN_BASE_URL,
    DEFAULT_CITY,
    REQUEST_TIMEOUT,
)

from src.utils.logger import logger


class AQICNClient:

    def __init__(self):
        self.api_key = AQICN_API_KEY
        self.base_url = AQICN_BASE_URL

    def _build_url(self, city: str) -> str:
        """
        Build AQICN request URL.
        """
        return f"{self.base_url}/{city}/?token={self.api_key}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        reraise=True,
    )
    def _request(self, city: str):

        url = self._build_url(city)

        logger.info(f"Fetching AQICN data for {city}")

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    def fetch_air_quality(self, city=None):

        city = city or DEFAULT_CITY

        response = self._request(city)

       # print("\n================ RAW AQICN RESPONSE ================\n")
       # print(response)
       # print("\n====================================================\n")

        if response.get("status") != "ok":

            logger.error(response)

            raise ValueError(
                f"""
AQICN API Error

Status : {response.get("status")}

Response:

{response}
"""
            )

        data = response["data"]

        iaqi = data.get("iaqi", {})

        normalized = {

            "city": data.get("city", {}).get("name"),

            "aqi": data.get("aqi"),

            "dominant_pollutant": data.get("dominentpol"),

            "pm25": iaqi.get("pm25", {}).get("v"),

            "pm10": iaqi.get("pm10", {}).get("v"),

            "temperature": iaqi.get("t", {}).get("v"),

            "humidity": iaqi.get("h", {}).get("v"),

            "pressure": iaqi.get("p", {}).get("v"),

            "wind_speed": iaqi.get("w", {}).get("v"),

            "dew_point": iaqi.get("dew", {}).get("v"),

            "timestamp": data.get("time", {}).get("iso"),

            "latitude": data.get("city", {}).get("geo", [None, None])[0],

            "longitude": data.get("city", {}).get("geo", [None, None])[1],

            "forecast": data.get("forecast", {}),

        }

        logger.success("AQICN data fetched successfully.")

        return normalized


if __name__ == "__main__":

    client = AQICNClient()

    result = client.fetch_air_quality()

    print("\n========== NORMALIZED DATA ==========\n")

    for key, value in result.items():
        print(f"{key:22}: {value}")