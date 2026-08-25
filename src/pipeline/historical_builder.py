"""
Historical Dataset Builder
--------------------------

Reads all raw JSON files from data/raw,
combines them into one dataset,
removes duplicates,
sorts by collection time,
and saves a clean CSV.

Author: Ahmad
Project: AirSense-AI
"""

import json
from pathlib import Path

import pandas as pd

from src.utils.logger import logger


class HistoricalDatasetBuilder:

    def __init__(self):

        self.raw_dir = Path("data/raw")

        self.output_dir = Path("data/processed")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_dir / "historical_data.csv"

    def build(self):

        logger.info("Reading raw JSON files...")

        json_files = sorted(self.raw_dir.glob("*.json"))

        if not json_files:

            logger.warning("No JSON files found.")

            return None

        records = []

        for file in json_files:

            try:

                with open(file, "r", encoding="utf-8") as f:

                    records.append(json.load(f))

            except Exception as e:

                logger.error(f"Failed to read {file.name}: {e}")

        df = pd.DataFrame(records)

        if df.empty:

            logger.warning("No valid data found.")

            return None

        if "collection_time" in df.columns:

            df["collection_time"] = pd.to_datetime(df["collection_time"])

            df = df.sort_values("collection_time")

        df = df.drop_duplicates()

        df.to_csv(self.output_file, index=False)

        logger.success(f"Historical dataset saved -> {self.output_file}")

        logger.info(f"Total records : {len(df)}")

        logger.info(f"Total columns : {len(df.columns)}")

        return df


if __name__ == "__main__":

    builder = HistoricalDatasetBuilder()

    df = builder.build()

    if df is not None:

        print("\n========== DATASET PREVIEW ==========\n")

        print(df.head())