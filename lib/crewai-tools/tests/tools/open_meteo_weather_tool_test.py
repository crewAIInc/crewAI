from unittest.mock import MagicMock, patch
from crewai_tools.tools.open_meteo_weather_tool.open_meteo_weather_tool import OpenMeteoWeatherTool


@patch("requests.get")
def test_open_meteo_weather_tool_success(mock_get):
    # Mock Geocoding API response
    mock_geo = MagicMock(status_code=200)
    mock_geo.json.return_value = {
        "results": [{"name": "Tokyo", "country": "Japan", "latitude": 35.6895, "longitude": 139.6917}]
    }

    # Mock Forecast API response
    mock_weather = MagicMock(status_code=200)
    mock_weather.json.return_value = {
        "current_weather": {"temperature": 18.5, "windspeed": 12.0}
    }

    mock_get.side_effect = [mock_geo, mock_weather]

    tool = OpenMeteoWeatherTool()
    result = tool._run(city_name="Tokyo")

    assert "Tokyo, Japan" in result
    assert "18.5°C" in result
    assert "12.0 km/h" in result


@patch("requests.get")
def test_open_meteo_weather_tool_city_not_found(mock_get):
    mock_geo = MagicMock(status_code=200)
    mock_geo.json.return_value = {"results": []}
    mock_get.return_value = mock_geo

    tool = OpenMeteoWeatherTool()
    result = tool._run(city_name="UnknownCity12345")

    assert "not found" in result
