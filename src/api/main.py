from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.prediction.forecast_engine import ForecastEngine


app = FastAPI(
    title="AirSense-AI API",
    description="PM2.5 forecasting and AQI inference service",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AirSense-AI",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/forecast")
def forecast():
    """
    Generate the production 72-hour forecast.

    Features are obtained through the ForecastEngine's
    production Feature Store path.
    """
    try:
        engine = ForecastEngine()
        result = engine.forecast()

        return JSONResponse(
            content={
                "status": "success",
                "forecast_hours": len(result),
                "forecast": result.to_dict(
                    orient="records"
                ),
            }
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "message": str(exc),
                "reason": (
                    "A valid production forecasting "
                    "window is currently unavailable."
                ),
            },
        ) from exc