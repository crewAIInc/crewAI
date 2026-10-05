from unittest.mock import MagicMock, patch

import requests

from crewai_tools import OpenMeteoWeatherTool


@patch("requests.get")
def test_open_meteo_weather_tool_success(mock_get):
    mock_geo = MagicMock(status_code=200)
    mock_geo.json.return_value = {
        "results": [
            {
                "name": "Tokyo",
                "country": "Japan",
                "latitude": 35.6895,
                "longitude": 139.6917,
            }
        ]
    }

    mock_weather = MagicMock(status_code=200)
    mock_weather.json.return_value = {
        "current": {
            "temperature_2m": 18.5,
            "wind_speed_10m": 12.0,
        }
    }

    mock_get.side_effect = [mock_geo, mock_weather]

    tool = OpenMeteoWeatherTool()
    result = tool._run(city_name="Tokyo")

    assert "Tokyo, Japan" in result
    assert "18.5°C" in result
    assert "12.0 km/h" in result
    assert "Weather data by Open-Meteo.com" in result


@patch("requests.get")
def test_open_meteo_weather_tool_city_not_found(mock_get):
    mock_geo = MagicMock(status_code=200)
    mock_geo.json.return_value = {"results": []}

    mock_get.return_value = mock_geo

    tool = OpenMeteoWeatherTool()
    result = tool._run(city_name="UnknownCity12345")

    assert "not found" in result


@patch("requests.get")
def test_open_meteo_weather_tool_empty_city(mock_get):
    tool = OpenMeteoWeatherTool()

    result = tool._run(city_name="   ")

    assert result == "Error: City name must be a non-empty string."
    mock_get.assert_not_called()


@patch("requests.get")
def test_open_meteo_weather_tool_geocoding_error(mock_get):
    mock_get.side_effect = requests.RequestException("service unavailable")

    tool = OpenMeteoWeatherTool()
    result = tool._run(city_name="Tokyo")

    assert "Error retrieving weather data for 'Tokyo'" in result
    assert "service unavailable" in result


@patch("requests.get")
def test_open_meteo_weather_tool_forecast_error(mock_get):
    mock_geo = MagicMock(status_code=200)
    mock_geo.json.return_value = {
        "results": [
            {
                "name": "Tokyo",
                "country": "Japan",
                "latitude": 35.6895,
                "longitude": 139.6917,
            }
        ]
    }

    mock_get.side_effect = [
        mock_geo,
        requests.Timeout("forecast request timed out"),
    ]

    tool = OpenMeteoWeatherTool()
    result = tool._run(city_name="Tokyo")

    assert "Error retrieving weather data for 'Tokyo'" in result
    assert "forecast request timed out" in result


@patch("requests.get")
def test_open_meteo_weather_tool_uses_current_weather_parameters(mock_get):
    mock_geo = MagicMock(status_code=200)
    mock_geo.json.return_value = {
        "results": [
            {
                "name": "Tokyo",
                "country": "Japan",
                "latitude": 35.6895,
                "longitude": 139.6917,
            }
        ]
    }

    mock_weather = MagicMock(status_code=200)
    mock_weather.json.return_value = {
        "current": {
            "temperature_2m": 18.5,
            "wind_speed_10m": 12.0,
        }
    }

    mock_get.side_effect = [mock_geo, mock_weather]

    tool = OpenMeteoWeatherTool()
    tool._run(city_name="Tokyo")

    assert mock_get.call_count == 2

    _, geo_kwargs = mock_get.call_args_list[0]
    assert geo_kwargs["params"]["name"] == "Tokyo"

    _, weather_kwargs = mock_get.call_args_list[1]
    assert weather_kwargs["params"]["current"] == ("temperature_2m,wind_speed_10m")
