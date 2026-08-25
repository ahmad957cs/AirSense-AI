from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")

SENSORS = [
    (25066, "US Diplomatic Post: Islamabad"),
    (1343270, "Islamabad"),
    (12960851, "Buraq integrated solutions"),
    (13137731, "F-7, Islamabad"),
    (13181992, "Rawalpindi Women University"),
    (13183766, "Wasa 3, OHR Office"),
    (13190424, "Social Welfare Department"),
    (13190993, "WASA Head Office 1"),
    (13191310, "Punjab Food Authority"),
    (13191604, "Public Health Engineering"),
    (13191609, "Conservator of Forest, North zone"),
    (13192453, "PHA Head Office"),
    (13192751, "E11/4 Sector Islamabad"),
    (13201383, "WASA 2"),
    (13229202, "Park Road F-8/1"),
    (13236697, "District Health Authority"),
    (13451278, "Govt. Graduate College"),
    (13493758, "NASTP"),
    (14671472, "Islamabad Station"),
    (14671479, "Street 8, F-7/3"),
    (14671482, "Dr Abuzar - Bahria Town"),
    (14671485, "Bahria Town Phase 2"),
    (14671502, "Mughal Village"),
    (14671507, "Islamabad H11 A"),
    (14671510, "Street #1 F-6/3"),
    (14671513, "F7-2 Street 20"),
    (14671520, "House Street22 E7"),
    (14671527, "Westridge1"),
    (14671534, "Rawalpindi A GT Road"),
    (14683997, "Device 1 Suraj Gali"),
    (14680875, "NASTP"),
    (14686629, "PHA Head Office"),
    (14686636, "Punjab Food Authority"),
    (14673718, "Fair Finance Pakistan Islamabad"),
    (14719910, "WASA 2 Sufaid tanki"),
    (13784055, "DC Office Sialkot"),
    (13817565, "DC office"),
]

URL_TEMPLATE = (
    "https://api.openaq.org/v3/sensors/{}/hours"
)

RECENT_DAYS = 30
PAGE_LIMIT = 100
MAX_PAGES = 100

OUTPUT = "data/processed/sensor_coverage.csv"


# ============================================================
# TIME WINDOW
# ============================================================

END_TIME = datetime.now(
    timezone.utc
)

START_TIME = (
    END_TIME
    - timedelta(days=RECENT_DAYS)
)

START = START_TIME.isoformat()
END = END_TIME.isoformat()


# ============================================================
# API
# ============================================================

headers = {
    "X-API-Key": API_KEY
}


def fetch_sensor_records(
    sensor_id: int,
) -> list[dict]:

    records = []

    page = 1

    while page <= MAX_PAGES:

        params = {
            "datetime_from": START,
            "datetime_to": END,
            "limit": PAGE_LIMIT,
            "page": page,
        }

        response = requests.get(
            URL_TEMPLATE.format(
                sensor_id
            ),
            headers=headers,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        page_records = payload.get(
            "results",
            []
        )

        if not page_records:
            break

        records.extend(
            page_records
        )

        if len(page_records) < PAGE_LIMIT:
            break

        page += 1

        time.sleep(0.15)

    return records


# ============================================================
# CONTINUITY
# ============================================================

def calculate_continuity(
    records: list[dict],
) -> dict:

    timestamps = []

    for item in records:

        value = (
            item
            .get("period", {})
            .get("datetimeFrom", {})
            .get("utc")
        )

        if value:
            timestamps.append(value)

    if not timestamps:

        return {
            "records": 0,
            "first": None,
            "last": None,
            "longest_block_hours": 0,
            "hours_available": 0,
        }

    ts = pd.to_datetime(
        pd.Series(timestamps),
        utc=True,
        errors="coerce",
    )

    ts = (
        ts.dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if ts.empty:

        return {
            "records": 0,
            "first": None,
            "last": None,
            "longest_block_hours": 0,
            "hours_available": 0,
        }

    gaps = ts.diff()

    groups = (
        gaps
        .ne(pd.Timedelta(hours=1))
        .cumsum()
    )

    blocks = (
        ts.groupby(groups)
        .agg(
            start="min",
            end="max",
            rows="size",
        )
    )

    blocks["hours"] = blocks["rows"]

    longest_block = int(
        blocks["hours"].max()
    )

    return {
        "records": int(len(ts)),
        "first": ts.min(),
        "last": ts.max(),
        "longest_block_hours":
            longest_block,
        "hours_available":
            int(len(ts)),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not API_KEY:
        raise ValueError(
            "OPENAQ_API_KEY is missing from .env"
        )

    print("=" * 70)
    print("OPENAQ RECENT SENSOR COVERAGE CHECK")
    print("=" * 70)

    print(
        f"Window: {START} → {END}"
    )

    results = []

    for index, (
        sensor_id,
        name,
    ) in enumerate(
        SENSORS,
        start=1,
    ):

        print(
            f"\n[{index}/{len(SENSORS)}] "
            f"{sensor_id} | {name}"
        )

        try:

            records = fetch_sensor_records(
                sensor_id
            )

            stats = calculate_continuity(
                records
            )

            result = {
                "sensor_id":
                    sensor_id,
                "name":
                    name,
                "records":
                    stats["records"],
                "first":
                    stats["first"],
                "last":
                    stats["last"],
                "longest_block_hours":
                    stats[
                        "longest_block_hours"
                    ],
                "hours_available":
                    stats[
                        "hours_available"
                    ],
                "has_72h_block":
                    stats[
                        "longest_block_hours"
                    ] >= 72,
                "status":
                    200,
            }

            results.append(result)

            print(
                f"Records: "
                f"{stats['records']}"
            )

            print(
                f"Latest: "
                f"{stats['last']}"
            )

            print(
                f"Longest continuous block: "
                f"{stats['longest_block_hours']} hours"
            )

            print(
                "72-hour block: "
                + (
                    "YES"
                    if stats[
                        "longest_block_hours"
                    ] >= 72
                    else "NO"
                )
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

            results.append(
                {
                    "sensor_id":
                        sensor_id,
                    "name":
                        name,
                    "records":
                        0,
                    "first":
                        None,
                    "last":
                        None,
                    "longest_block_hours":
                        0,
                    "hours_available":
                        0,
                    "has_72h_block":
                        False,
                    "status":
                        "ERROR",
                }
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    df["first"] = pd.to_datetime(
        df["first"],
        utc=True,
        errors="coerce",
    )

    df["last"] = pd.to_datetime(
        df["last"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Ranking
    #
    # Priority:
    # 1. Has a 72h continuous block
    # 2. Latest data
    # 3. Longest continuous block
    # 4. Recent record count
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "has_72h_block",
            "last",
            "longest_block_hours",
            "records",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RECENT SENSOR COVERAGE RANKING")
    print("=" * 70)

    print(
        df[
            [
                "sensor_id",
                "name",
                "records",
                "last",
                "longest_block_hours",
                "has_72h_block",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()