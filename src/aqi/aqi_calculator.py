from __future__ import annotations

from typing import Optional

import math


# ============================================================
# US EPA PM2.5 AQI breakpoints
#
# PM2.5 AQI is based on a 24-hour average concentration.
# Values below are current EPA breakpoint ranges.
# ============================================================

PM25_BREAKPOINTS = [
    {
        "aqi_low": 0,
        "aqi_high": 50,
        "conc_low": 0.0,
        "conc_high": 9.0,
        "category": "Good",
    },
    {
        "aqi_low": 51,
        "aqi_high": 100,
        "conc_low": 9.1,
        "conc_high": 35.4,
        "category": "Moderate",
    },
    {
        "aqi_low": 101,
        "aqi_high": 150,
        "conc_low": 35.5,
        "conc_high": 55.4,
        "category": "Unhealthy for Sensitive Groups",
    },
    {
        "aqi_low": 151,
        "aqi_high": 200,
        "conc_low": 55.5,
        "conc_high": 125.4,
        "category": "Unhealthy",
    },
    {
        "aqi_low": 201,
        "aqi_high": 300,
        "conc_low": 125.5,
        "conc_high": 225.4,
        "category": "Very Unhealthy",
    },
    {
        "aqi_low": 301,
        "aqi_high": 500,
        "conc_low": 225.5,
        "conc_high": 325.4,
        "category": "Hazardous",
    },
]


def truncate_pm25(value: float) -> float:
    """
    Truncate PM2.5 concentration to one decimal place.
    """
    if not math.isfinite(value):
        raise ValueError(
            "PM2.5 concentration must be finite."
        )

    if value < 0:
        value = 0.0

    return math.floor(value * 10) / 10


def pm25_to_aqi(
    pm25_24h: float,
) -> tuple[Optional[int], Optional[str]]:
    """
    Convert a 24-hour PM2.5 concentration to
    the corresponding US EPA AQI sub-index.

    Returns:
        (AQI, category)

    Returns (None, None) when the concentration is
    outside the supported range.
    """

    concentration = truncate_pm25(
        pm25_24h
    )

    for bp in PM25_BREAKPOINTS:

        if (
            bp["conc_low"]
            <= concentration
            <= bp["conc_high"]
        ):

            aqi = (
                (
                    bp["aqi_high"]
                    - bp["aqi_low"]
                )
                /
                (
                    bp["conc_high"]
                    - bp["conc_low"]
                )
            ) * (
                concentration
                - bp["conc_low"]
            ) + bp["aqi_low"]

            return (
                int(round(aqi)),
                bp["category"],
            )

    # AQI > 500 is outside the normal public index range.
    return None, None


def classify_aqi(aqi: float) -> str:
    """
    Classify an AQI value into its category.
    """

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"