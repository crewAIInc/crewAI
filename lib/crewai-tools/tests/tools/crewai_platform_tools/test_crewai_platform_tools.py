import unittest
from unittest.mock import Mock, patch

from crewai_tools.tools.crewai_platform_tools import CrewaiPlatformTools
from crewai_tools.tools.crewai_platform_tools import file_hook


class TestCrewaiPlatformTools(unittest.TestCase):
    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_crewai_platform_tools_basic(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {"github": []}}
        mock_get.return_value = mock_response

        tools = CrewaiPlatformTools(apps=["github"])
        assert tools is not None
        assert isinstance(tools, list)

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_crewai_platform_tools_multiple_apps(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "actions": {
                "github": [
                    {
                        "name": "create_issue",
                        "description": "Create a GitHub issue",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Issue title",
                                },
                                "body": {"type": "string", "description": "Issue body"},
                            },
                            "required": ["title"],
                        },
                    }
                ],
                "slack": [
                    {
                        "name": "send_message",
                        "description": "Send a Slack message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "channel": {
                                    "type": "string",
                                    "description": "Channel to send to",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Message text",
                                },
                            },
                            "required": ["channel", "text"],
                        },
                    }
                ],
            }
        }
        mock_get.return_value = mock_response

        tools = CrewaiPlatformTools(apps=["github", "slack"])
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) == 2

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert (
            "apps=github,slack" in args[0]
            or kwargs.get("params", {}).get("apps") == "github,slack"
        )

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    def test_crewai_platform_tools_empty_apps(self):
        with patch(
            "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
        ) as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"actions": {}}
            mock_get.return_value = mock_response

            tools = CrewaiPlatformTools(apps=[])
            assert tools is not None
            assert isinstance(tools, list)
            assert len(tools) == 0

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_crewai_platform_tools_api_error_handling(self, mock_get):
        mock_get.side_effect = Exception("API Error")

        tools = CrewaiPlatformTools(apps=["github"])
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) == 0

    def test_crewai_platform_tools_no_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                CrewaiPlatformTools(apps=["github"])
            assert "No platform integration token found" in str(context.exception)

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tools.register_file_processing_hook"
    )
    def test_crewai_platform_tools_registers_file_hook(
        self, mock_register_hook, mock_get
    ):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {"github": []}}
        mock_get.return_value = mock_response

        CrewaiPlatformTools(apps=["github"])
        mock_register_hook.assert_called_once()


class TestFileHook(unittest.TestCase):
    def setUp(self):
        file_hook._hook_registered = False

    def tearDown(self):
        file_hook._hook_registered = False

    @patch("crewai.hooks.register_after_tool_call_hook")
    def test_register_hook_is_idempotent(self, mock_register):
        """Test hook registration succeeds once and is idempotent."""
        assert file_hook.register_file_processing_hook() is True
        assert file_hook._hook_registered is True
        mock_register.assert_called_once_with(file_hook.process_file_markers)

        # Second call should return False and not register again
        assert file_hook.register_file_processing_hook() is False
        mock_register.assert_called_once()

    def test_process_file_markers_ignores_non_file_results(self):
        """Test that non-file-marker results return None."""
        test_cases = [
            None,  # Empty result
            "Regular tool output",  # Non-marker
            "__CREWAI_FILE__:incomplete",  # Invalid format (missing parts)
        ]
        for tool_result in test_cases:
            mock_context = Mock()
            mock_context.tool_result = tool_result
            assert file_hook.process_file_markers(mock_context) is None

    def test_format_file_size(self):
        """Test file size formatting across units."""
        cases = [
            (500, "500 bytes"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1024 * 1024, "1.0 MB"),
            (1024 * 1024 * 1024, "1.0 GB"),
        ]
        for size_bytes, expected in cases:
            assert file_hook._format_file_size(size_bytes) == expected
