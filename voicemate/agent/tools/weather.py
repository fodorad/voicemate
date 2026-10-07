"""Weather forecasts from Open-Meteo (free, no API key)."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from voicemate.agent.tools.base import ToolContext, ToolError, report

#: Open-Meteo forecast endpoint.
FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"

#: Open-Meteo geocoding endpoint.
GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"

#: WMO weather interpretation codes used by Open-Meteo.
WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    56: "freezing drizzle",
    57: "dense freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def describe(code: int | None) -> str:
    """Human-readable description of a WMO weather code."""
    if code is None:
        return "unknown"
    return WMO_CODES.get(int(code), f"weather code {code}")


def format_forecast(data: dict[str, Any], place: str) -> str:
    """Render an Open-Meteo response as compact text for the model.

    Args:
        data: JSON with ``current`` and ``daily`` blocks.
        place: Location name to mention.
    """
    lines = [f"Weather for {place}:"]
    current = data.get("current") or {}
    if current:
        lines.append(
            f"Now: {current.get('temperature_2m')}°C, {describe(current.get('weather_code'))}, "
            f"wind {current.get('wind_speed_10m')} km/h."
        )
    daily = data.get("daily") or {}
    for i, day in enumerate(daily.get("time", [])):
        lines.append(
            f"{day}: {describe(daily['weather_code'][i])}, "
            f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°C, "
            f"precipitation chance {daily['precipitation_probability_max'][i]}%."
        )
    return "\n".join(lines)


def build(ctx: ToolContext) -> list[BaseTool]:
    """Create the weather tool."""

    async def geocode(name: str) -> tuple[float, float, str]:
        response = await ctx.http.get(
            GEOCODING_URL, params={"name": name, "count": 1, "language": "en", "format": "json"}
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            raise ToolError(f"Unknown location {name!r}")
        hit = results[0]
        label = ", ".join(x for x in (hit.get("name"), hit.get("country")) if x)
        return float(hit["latitude"]), float(hit["longitude"]), label

    async def get_weather(config: RunnableConfig, location: str = "", days: int = 1) -> str:
        """Get current weather and the daily forecast.

        Args:
            location: Place name; leave empty for the user's home location.
            days: Forecast days including today (1-7); use 2 for "tomorrow".
        """
        days = max(1, min(7, int(days)))
        if location.strip():
            lat, lon, place = await geocode(location.strip())
        else:
            loc = ctx.config.location
            lat, lon, place = loc.latitude, loc.longitude, loc.name
        response = await ctx.http.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": days,
            },
        )
        response.raise_for_status()
        report(config, "get_weather", {"location": place, "days": days}, "web", f"weather {place}")
        return format_forecast(response.json(), place)

    return [
        StructuredTool.from_function(
            coroutine=get_weather, name="get_weather", parse_docstring=True
        )
    ]
