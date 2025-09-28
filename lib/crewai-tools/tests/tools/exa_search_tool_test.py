import os
from unittest.mock import patch

from crewai_tools import EXASearchTool
import pytest


@pytest.fixture
def exa_search_tool():
    return EXASearchTool(api_key="test_api_key")


@pytest.fixture(autouse=True)
def mock_exa_api_key():
    with patch.dict(os.environ, {"EXA_API_KEY": "test_key_from_env"}):
        yield


def test_exa_search_tool_initialization():
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "crewai_tools.tools.exa_tools.exa_search_tool.Exa"
        ) as mock_exa_class:
            api_key = "test_api_key"
            tool = EXASearchTool(api_key=api_key)

            assert tool.api_key == api_key
            assert tool.content is False
            assert tool.summary is False
            assert tool.type == "auto"
            mock_exa_class.assert_called_once_with(api_key=api_key)


def test_exa_search_tool_initialization_with_env(mock_exa_api_key):
    with patch.dict(os.environ, {"EXA_API_KEY": "test_key_from_env"}, clear=True):
        with patch(
            "crewai_tools.tools.exa_tools.exa_search_tool.Exa"
        ) as mock_exa_class:
            EXASearchTool()
            mock_exa_class.assert_called_once_with(api_key="test_key_from_env")


def test_exa_search_tool_initialization_with_base_url():
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "crewai_tools.tools.exa_tools.exa_search_tool.Exa"
        ) as mock_exa_class:
            api_key = "test_api_key"
            base_url = "https://custom.exa.api.com"
            tool = EXASearchTool(api_key=api_key, base_url=base_url)

            assert tool.api_key == api_key
            assert tool.base_url == base_url
            assert tool.content is False
            assert tool.summary is False
            assert tool.type == "auto"
            mock_exa_class.assert_called_once_with(api_key=api_key, base_url=base_url)


@pytest.fixture
def mock_exa_base_url():
    with patch.dict(os.environ, {"EXA_BASE_URL": "https://env.exa.api.com"}):
        yield


def test_exa_search_tool_initialization_with_env_base_url(
    mock_exa_api_key, mock_exa_base_url
):
    with patch("crewai_tools.tools.exa_tools.exa_search_tool.Exa") as mock_exa_class:
        EXASearchTool()
        mock_exa_class.assert_called_once_with(
            api_key="test_key_from_env", base_url="https://env.exa.api.com"
        )


def test_exa_search_tool_initialization_without_base_url():
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "crewai_tools.tools.exa_tools.exa_search_tool.Exa"
        ) as mock_exa_class:
            api_key = "test_api_key"
            tool = EXASearchTool(api_key=api_key)

            assert tool.api_key == api_key
            assert tool.base_url is None
            mock_exa_class.assert_called_once_with(api_key=api_key)
