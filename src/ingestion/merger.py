"""
Data Merger
-----------

Merge AQICN and OpenWeather data into one unified record.

Author: Ahmad
Project: AirSense-AI
"""

from datetime import datetime, UTC

from src.ingestion.aqicn import AQICNClient
from src.ingestion.openweather import OpenWeatherClient

from src.utils.logger import logger


class DataMerger:
    """
    Merge multiple data sources into one dictionary.
    """

    def __init__(self):

        self.aqicn = AQICNClient()

        self.weather = OpenWeatherClient()

    def merge(self, city=None):

        logger.info("Fetching data from all sources...")

        aqi = self.aqicn.fetch_air_quality(city)

        weather = self.weather.fetch_weather(city)

        merged = {

            # Metadata
            "collection_time": datetime.now(UTC).isoformat(),

            "city": weather["city"],

            "country": weather["country"],

            "latitude": weather["latitude"],

            "longitude": weather["longitude"],

            # AQI
            "aqi": aqi["aqi"],

            "dominant_pollutant": aqi["dominant_pollutant"],

            "pm25": aqi["pm25"],

            "pm10": aqi["pm10"],

            # Weather
            "temperature": weather["temperature"],

            "feels_like": weather["feels_like"],

            "humidity": weather["humidity"],

            "pressure": weather["pressure"],

            "visibility": weather["visibility"],

            "wind_speed": weather["wind_speed"],

            "wind_degree": weather["wind_degree"],

            "wind_gust": weather["wind_gust"],

            "cloud_cover": weather["cloud_cover"],

            "weather": weather["weather"],

            "weather_description": weather["weather_description"],

            "sunrise": weather["sunrise"],

            "sunset": weather["sunset"],

            "timezone": weather["timezone"],

            "api_timestamp": weather["timestamp"]

        }

        logger.success("Data merged successfully.")

        return merged


if __name__ == "__main__":

    merger = DataMerger()

    data = merger.merge()

    print("\n========== MERGED DATA ==========\n")

    for key, value in data.items():

        print(f"{key:25}: {value}")