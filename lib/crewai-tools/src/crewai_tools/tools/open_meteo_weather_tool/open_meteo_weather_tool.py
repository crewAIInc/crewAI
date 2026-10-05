
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


class OpenMeteoWeatherToolInput(BaseModel):
    """Input schema for OpenMeteoWeatherTool."""

    city_name: str = Field(
        description="The name of the city to query weather for (e.g. 'San Francisco', 'Tokyo', 'London')."
    )


class OpenMeteoWeatherTool(BaseTool):
    name: str = "Open-Meteo Weather"
    description: str = (
        "Retrieves live weather data (temperature, wind speed, weather conditions) "
        "for a specified city using the free Open-Meteo REST API."
    )
    args_schema: type[BaseModel] = OpenMeteoWeatherToolInput

    def _run(self, city_name: str) -> str:
        """Fetch current weather for a specified city name."""
        if not city_name or not city_name.strip():
            return "Error: City name must be a non-empty string."

        city = city_name.strip()
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"

        try:
            # Step 1: Geocode city name to lat/lon
            geo_res = requests.get(geocoding_url, timeout=10)
            geo_res.raise_for_status()
            geo_data = geo_res.json()

            results = geo_data.get("results")
            if not results or not isinstance(results, list):
                return f"Location '{city}' not found."

            location = results[0]
            lat = location.get("latitude")
            lon = location.get("longitude")
            name = location.get("name", city)
            country = location.get("country", "")

            # Step 2: Fetch current weather for coordinates
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            weather_res = requests.get(weather_url, timeout=10)
            weather_res.raise_for_status()
            weather_data = weather_res.json()

            current = weather_data.get("current_weather", {})
            temp = current.get("temperature", "N/A")
            windspeed = current.get("windspeed", "N/A")

            location_str = f"{name}, {country}" if country else name
            return (
                f"Current Weather for {location_str}:\n"
                f"- Temperature: {temp}°C\n"
                f"- Wind Speed: {windspeed} km/h"
            )

        except Exception as e:
            return f"Error retrieving weather data for '{city}': {e!s}"
