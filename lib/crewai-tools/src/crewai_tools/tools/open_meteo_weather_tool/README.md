# Open-Meteo Weather Tool

The `OpenMeteoWeatherTool` allows CrewAI agents to retrieve current weather information for a city using the public Open-Meteo REST APIs.

## Features

- No API key or account is required for the Open-Meteo free API.
- Converts a city name to coordinates using Open-Meteo's Geocoding API.
- Retrieves current temperature and wind speed.
- Handles unknown locations and HTTP failures gracefully.

## Usage

```python
from crewai_tools import OpenMeteoWeatherTool

tool = OpenMeteoWeatherTool()

result = tool.run(city_name="Tokyo")
print(result)
