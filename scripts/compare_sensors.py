import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAQ_API_KEY")

headers = {
    "X-API-Key": key
}

sensors = {
    "13137731": "F-7 Islamabad",
    "14719910": "WASA 2 Sufaid tanki"
}

start = "2025-01-01T00:00:00Z"
end = "2026-08-10T00:00:00Z"

print("=" * 60)
print("FINAL SENSOR COMPARISON")
print("=" * 60)

for sensor_id, name in sensors.items():

    print()
    print("Checking:", name)
    print("Sensor ID:", sensor_id)

    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"

    page = 1
    rows = []

    while True:

        params = {
            "datetime_from": start,
            "datetime_to": end,
            "limit": 100,
            "page": page
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        rows.extend(results)

        print(f"Page {page}: {len(results)} records")

        page += 1

        if len(results) < 100:
            break

    print()
    print("===== RESULT =====")

    if not rows:
        print("No data found.")
        continue

    timestamps = pd.to_datetime(
        [
            row["period"]["datetimeFrom"]["utc"]
            for row in rows
        ],
        utc=True
    ).sort_values()

    gaps = timestamps.to_series().diff().dropna()

    one_hour = pd.Timedelta(hours=1)

    exact_one_hour = int(
        (gaps == one_hour).sum()
    )

    missing_hours = int(
        ((gaps / one_hour) - 1)
        .clip(lower=0)
        .sum()
    )

    largest_gap = gaps.max()

    print("Sensor:", sensor_id)
    print("Location:", name)
    print("Records:", len(rows))
    print("First:", timestamps.min())
    print("Last:", timestamps.max())
    print("1-hour intervals:", exact_one_hour)
    print("Missing hourly slots:", missing_hours)
    print("Largest gap:", largest_gap)

    print("-" * 60)

print()
print("=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)