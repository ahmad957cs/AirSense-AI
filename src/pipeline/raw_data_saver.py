"""
Raw Data Saver
--------------

Fetch merged AQI and weather data and save it as a timestamped JSON file.

Author: Ahmad
Project: AirSense-AI
"""

import json
from datetime import datetime, UTC
from pathlib import Path

from src.ingestion.merger import DataMerger
from src.utils.logger import logger


class RawDataSaver:
    """
    Save merged API data into the raw data directory.
    """

    def __init__(self):

        self.output_dir = Path("data/raw")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.merger = DataMerger()

    def save(self, city=None):

        logger.info("Collecting merged data...")

        data = self.merger.merge(city)

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
            )

        logger.success(f"Raw data saved -> {filepath}")

        return filepath


if __name__ == "__main__":

    saver = RawDataSaver()

    file = saver.save()

    print("\nSaved file:\n")

    print(file)