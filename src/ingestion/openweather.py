"""
OpenWeather API Client
----------------------

Fetches current weather data from OpenWeather API.

Author: Ahmad
Project: AirSense-AI

Responsibilities
----------------
1. Connect to OpenWeather API
2. Handle retries
3. Validate API responses
4. Normalize weather data
5. Return clean Python dictionary
"""

import requests

from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_fixed

from config.config import (
    OPENWEATHER_API_KEY,
    OPENWEATHER_BASE_URL,
    DEFAULT_CITY,
    REQUEST_TIMEOUT,
)

from src.utils.logger import logger


class OpenWeatherClient:
    """
    Client for OpenWeather API.
    """

    BASE_URL = OPENWEATHER_BASE_URL

    def __init__(self):

        self.api_key = OPENWEATHER_API_KEY

    def _build_params(self, city: str):

        return {

            "q": city,

            "appid": self.api_key,

            "units": "metric",

        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        reraise=True,
    )
    def _request(self, city: str):

        logger.info(f"Fetching OpenWeather data for {city}")

        response = requests.get(

            OPENWEATHER_BASE_URL,

            params=self._build_params(city),

            timeout=REQUEST_TIMEOUT,

        )

        response.raise_for_status()

        return response.json()

    def fetch_weather(self, city: str | None = None) -> dict:
        city = city or DEFAULT_CITY
        data = self._request(city)

        if data.get("cod") != 200:
            raise ValueError(
                f"OpenWeather returned an invalid response: {data}"
            )

        weather = {

            "city": data.get("name"),

            "country": data.get("sys", {}).get("country"),

            "latitude": data.get("coord", {}).get("lat"),

            "longitude": data.get("coord", {}).get("lon"),

            "temperature": data.get("main", {}).get("temp"),

            "feels_like": data.get("main", {}).get("feels_like"),

            "temp_min": data.get("main", {}).get("temp_min"),

            "temp_max": data.get("main", {}).get("temp_max"),

            "humidity": data.get("main", {}).get("humidity"),

            "pressure": data.get("main", {}).get("pressure"),

            "sea_level": data.get("main", {}).get("sea_level"),

            "ground_level": data.get("main", {}).get("grnd_level"),

            "visibility": data.get("visibility"),

            "wind_speed": data.get("wind", {}).get("speed"),

            "wind_degree": data.get("wind", {}).get("deg"),

            "wind_gust": data.get("wind", {}).get("gust"),

            "cloud_cover": data.get("clouds", {}).get("all"),

            "weather": data.get("weather", [{}])[0].get("main"),

            "weather_description": data.get("weather", [{}])[0].get("description"),

            "weather_icon": data.get("weather", [{}])[0].get("icon"),

            "sunrise": data.get("sys", {}).get("sunrise"),

            "sunset": data.get("sys", {}).get("sunset"),

            "timezone": data.get("timezone"),

            "timestamp": data.get("dt"),

        }

        logger.success("OpenWeather data fetched successfully.")

        return weather


if __name__ == "__main__":

    client = OpenWeatherClient()

    result = client.fetch_weather()

    print("\n========== OPENWEATHER DATA ==========\n")

    for key, value in result.items():
        print(f"{key:22}: {value}")