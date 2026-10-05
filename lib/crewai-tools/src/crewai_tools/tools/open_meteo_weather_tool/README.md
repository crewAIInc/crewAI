# Open-Meteo Weather Tool

The `OpenMeteoWeatherTool` enables CrewAI agents to retrieve current weather conditions (temperature, wind speed, and general weather) for any city globally using the free, open-source [Open-Meteo REST API](https://open-meteo.com/).

It requires no API keys, accounts, or authentication.

---

## Features

- **Zero Configuration:** Works out of the box with zero API keys or environment variables required.
- **Automatic Geocoding:** Converts human-readable city names into geographic coordinates behind the scenes.
- **Lightweight & Fast:** Uses direct HTTP calls with no heavy external SDK dependencies.

---

## Installation & Requirements

Included natively in `crewai-tools`:

```bash
pip install crewai-tools
```

Dependencies:
- `requests`
- `pydantic`
- `crewai`

---

## Usage

### Standalone Usage

```python
from crewai_tools import OpenMeteoWeatherTool

# Initialize tool
weather_tool = OpenMeteoWeatherTool()

# Retrieve weather for a specific city
result = weather_tool.run(city_name="Tokyo")
print(result)
```

### Integration with CrewAI Agent

```python
from crewai import Agent, Task, Crew
from crewai_tools import OpenMeteoWeatherTool

weather_tool = OpenMeteoWeatherTool()

travel_agent = Agent(
    role="Travel & Itinerary Coordinator",
    goal="Provide up-to-date travel insights and local weather forecasts for travelers.",
    backstory="An AI logistics expert skilled at preparing travelers for local conditions.",
    tools=[weather_tool],
    verbose=True,
)

task = Task(
    description="Check current weather conditions in Tokyo and recommend suitable attire.",
    expected_output="A concise summary of current weather in Tokyo with practical clothing advice.",
    agent=travel_agent,
)

crew = Crew(agents=[travel_agent], tasks=[task])
crew.kickoff()
```

---

## Input Schema

The tool accepts parameters defined by `OpenMeteoWeatherToolInput`:

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `city_name` | `str` | Yes | Name of the city to query (e.g. `"San Francisco"`, `"Tokyo"`, `"London"`). |

---

## Example Output

```text
Current Weather for Tokyo, Japan:
- Temperature: 18.5°C
- Wind Speed: 12.0 km/h
```
