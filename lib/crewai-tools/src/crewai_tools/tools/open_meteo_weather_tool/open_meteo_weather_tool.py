from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


class OpenMeteoWeatherToolInput(BaseModel):
    """Input schema for OpenMeteoWeatherTool."""

    city_name: str = Field(
        description=(
            "The name of the city to query weather for "
            "(e.g. 'San Francisco', 'Tokyo', 'London')."
        )
    )


class OpenMeteoWeatherTool(BaseTool):
    """Retrieve current weather information for a city."""

    name: str = "Open-Meteo Weather"
    description: str = (
        "Retrieves current temperature and wind speed for a specified city "
        "using the public Open-Meteo REST API."
    )
    args_schema: type[BaseModel] = OpenMeteoWeatherToolInput

    def _run(self, city_name: str) -> str:
        """Fetch current weather for a specified city name."""
        if not city_name or not city_name.strip():
            return "Error: City name must be a non-empty string."

        city = city_name.strip()

        try:
            geo_res = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": city,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
                timeout=10,
            )
            geo_res.raise_for_status()

            geo_data = geo_res.json()

            if not isinstance(geo_data, dict):
                return f"Location '{city}' not found."

            results = geo_data.get("results")

            if (
                not isinstance(results, list)
                or not results
                or not isinstance(results[0], dict)
            ):
                return f"Location '{city}' not found."

            location = results[0]
            lat = location.get("latitude")
            lon = location.get("longitude")

            if lat is None or lon is None:
                return f"Location '{city}' not found."

            name = location.get("name", city)
            country = location.get("country", "")

            weather_res = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,wind_speed_10m",
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                },
                timeout=10,
            )
            weather_res.raise_for_status()

            weather_data = weather_res.json()

            if not isinstance(weather_data, dict):
                return "Error: Unexpected response from the weather API."

            current = weather_data.get("current")

            if not isinstance(current, dict):
                return "Error: Current weather data is unavailable."

            temperature = current.get("temperature_2m", "N/A")
            wind_speed = current.get("wind_speed_10m")
            if wind_speed is None:
                wind_speed = "N/A"

            location_str = f"{name}, {country}" if country else name

            return (
                f"Current Weather for {location_str}:\n"
                f"- Temperature: {temperature}°C\n"
                f"- Wind Speed: {wind_speed} km/h\n"
                "- Source: Weather data by Open-Meteo.com "
                "(https://open-meteo.com/), licensed under CC BY 4.0 "
                "(https://creativecommons.org/licenses/by/4.0/)."
            )

        except requests.RequestException as exc:
            return f"Error retrieving weather data for '{city}': {exc}"
        except (ValueError, TypeError, KeyError) as exc:
            return f"Error retrieving weather data for '{city}': {exc}"
