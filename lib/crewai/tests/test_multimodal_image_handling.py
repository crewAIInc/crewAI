"""Tests for multimodal image handling fixes (Issue #4016).

This module tests the fixes for proper multimodal image handling in CrewAI:
1. CrewAgentExecutor._handle_agent_action properly appends multimodal messages
2. ToolUsage.use preserves raw dict result for add_image tool
3. Gemini provider properly converts image_url to Gemini's format
"""

import base64
from unittest.mock import MagicMock, patch

import pytest


class TestCrewAgentExecutorMultimodal:
    """Tests for CrewAgentExecutor._handle_agent_action with multimodal content."""

    def test_handle_agent_action_with_add_image_tool_dict_result(self):
        """Test that add_image tool result dict is appended directly without wrapping."""
        from crewai.agents.crew_agent_executor import CrewAgentExecutor
        from crewai.agents.parser import AgentAction
        from crewai.tools.tool_usage import ToolResult

        # Create a mock executor
        mock_llm = MagicMock()
        mock_task = MagicMock()
        mock_crew = MagicMock()
        mock_agent = MagicMock()
        mock_agent.i18n = MagicMock()
        mock_agent.i18n.tools.return_value = {"name": "Add image to content"}

        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt="test prompt",
            tools=[],
            original_tools=[],
            max_iter=1,
        )
        executor.messages = []
        executor._i18n = mock_agent.i18n

        # Create a mock add_image tool result (dict with role and content)
        multimodal_result = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }

        formatted_answer = AgentAction(
            tool="Add image to content",
            tool_input='{"image_url": "https://example.com/image.jpg"}',
            text="Using add image tool",
            thought="I need to add an image",
            result="",
        )

        tool_result = ToolResult(result=multimodal_result)

        # Call _handle_agent_action
        result = executor._handle_agent_action(formatted_answer, tool_result)

        # Verify the message was appended correctly (not double-wrapped)
        assert len(executor.messages) == 1
        appended_message = executor.messages[0]

        # The message should be the dict directly, not wrapped
        assert appended_message["role"] == "user"
        assert isinstance(appended_message["content"], list)
        assert len(appended_message["content"]) == 2
        assert appended_message["content"][0]["type"] == "text"
        assert appended_message["content"][1]["type"] == "image_url"

    def test_handle_agent_action_with_add_image_tool_list_result(self):
        """Test that add_image tool result list is wrapped with user role."""
        from crewai.agents.crew_agent_executor import CrewAgentExecutor
        from crewai.agents.parser import AgentAction
        from crewai.tools.tool_usage import ToolResult

        # Create a mock executor
        mock_llm = MagicMock()
        mock_task = MagicMock()
        mock_crew = MagicMock()
        mock_agent = MagicMock()
        mock_agent.i18n = MagicMock()
        mock_agent.i18n.tools.return_value = {"name": "Add image to content"}

        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=mock_task,
            crew=mock_crew,
            agent=mock_agent,
            prompt="test prompt",
            tools=[],
            original_tools=[],
            max_iter=1,
        )
        executor.messages = []
        executor._i18n = mock_agent.i18n

        # Create a mock add_image tool result (list without role)
        content_list = [
            {"type": "text", "text": "Analyze this image"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]

        formatted_answer = AgentAction(
            tool="Add image to content",
            tool_input='{"image_url": "https://example.com/image.jpg"}',
            text="Using add image tool",
            thought="I need to add an image",
            result="",
        )

        tool_result = ToolResult(result=content_list)

        # Call _handle_agent_action
        result = executor._handle_agent_action(formatted_answer, tool_result)

        # Verify the message was wrapped with user role
        assert len(executor.messages) == 1
        appended_message = executor.messages[0]

        assert appended_message["role"] == "user"
        assert appended_message["content"] == content_list


class TestToolUsageMultimodal:
    """Tests for ToolUsage.use with add_image tool."""

    def test_tool_usage_preserves_add_image_dict_result(self):
        """Test that add_image tool result is not stringified."""
        from crewai.tools.tool_usage import ToolUsage

        # Create mock components
        mock_tools_handler = MagicMock()
        mock_tools_handler.cache = None
        mock_task = MagicMock()
        mock_task.used_tools = 0
        mock_agent = MagicMock()
        mock_agent.key = "test_key"
        mock_agent.role = "test_role"
        mock_agent.verbose = False
        mock_agent.fingerprint = None
        mock_agent.tools_results = []
        mock_action = MagicMock()
        mock_action.tool = "Add image to content"
        mock_action.tool_input = '{"image_url": "https://example.com/image.jpg"}'

        # Mock i18n
        mock_i18n = MagicMock()
        mock_i18n.tools.return_value = {"name": "Add image to content"}

        # Create a mock add_image tool
        mock_tool = MagicMock()
        mock_tool.name = "Add image to content"
        mock_tool.args_schema = MagicMock()
        mock_tool.args_schema.model_json_schema.return_value = {
            "properties": {"image_url": {}, "action": {}}
        }

        # The tool returns a dict with role and content
        multimodal_result = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }
        mock_tool.invoke.return_value = multimodal_result

        tool_usage = ToolUsage(
            tools_handler=mock_tools_handler,
            tools=[mock_tool],
            task=mock_task,
            function_calling_llm=MagicMock(),
            agent=mock_agent,
            action=mock_action,
        )
        tool_usage._i18n = mock_i18n
        tool_usage._telemetry = MagicMock()

        # Create a mock calling object
        mock_calling = MagicMock()
        mock_calling.tool_name = "Add image to content"
        mock_calling.arguments = {"image_url": "https://example.com/image.jpg"}

        # Call _use directly
        result = tool_usage._use(
            tool_string="test",
            tool=mock_tool,
            calling=mock_calling,
        )

        # The result should be the dict, not a string
        assert isinstance(result, dict)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)


class TestGeminiMultimodalFormatting:
    """Tests for Gemini provider's multimodal content handling."""

    @pytest.fixture
    def mock_gemini_types(self):
        """Create mock Gemini types for testing."""
        mock_types = MagicMock()
        mock_part = MagicMock()
        mock_content = MagicMock()

        # Mock Part.from_text
        mock_types.Part.from_text.return_value = mock_part

        # Mock Part.from_bytes
        mock_types.Part.from_bytes.return_value = mock_part

        # Mock Content
        mock_types.Content.return_value = mock_content

        return mock_types

    def test_convert_content_to_parts_simple_text(self):
        """Test converting simple text content to Gemini Parts."""
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": MagicMock(),
                "google.genai.types": MagicMock(),
                "google.genai.errors": MagicMock(),
            },
        ):
            from crewai.llms.providers.gemini.completion import GeminiCompletion

            # Create a mock instance
            completion = MagicMock(spec=GeminiCompletion)
            completion._convert_content_to_parts = (
                GeminiCompletion._convert_content_to_parts.__get__(
                    completion, GeminiCompletion
                )
            )

            # Mock types.Part.from_text
            mock_part = MagicMock()
            with patch(
                "crewai.llms.providers.gemini.completion.types.Part.from_text",
                return_value=mock_part,
            ):
                result = completion._convert_content_to_parts("Hello, world!")

            assert len(result) == 1

    def test_convert_content_to_parts_multimodal_with_image_url(self):
        """Test converting multimodal content with image_url to Gemini Parts."""
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": MagicMock(),
                "google.genai.types": MagicMock(),
                "google.genai.errors": MagicMock(),
            },
        ):
            from crewai.llms.providers.gemini.completion import GeminiCompletion

            # Create a mock instance
            completion = MagicMock(spec=GeminiCompletion)
            completion._convert_content_to_parts = (
                GeminiCompletion._convert_content_to_parts.__get__(
                    completion, GeminiCompletion
                )
            )
            completion._create_image_part_from_url = MagicMock(
                return_value=MagicMock()
            )

            # Mock types.Part.from_text
            mock_text_part = MagicMock()
            with patch(
                "crewai.llms.providers.gemini.completion.types.Part.from_text",
                return_value=mock_text_part,
            ):
                multimodal_content = [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ]

                result = completion._convert_content_to_parts(multimodal_content)

            # Should have called _create_image_part_from_url
            completion._create_image_part_from_url.assert_called_once_with(
                "https://example.com/image.jpg"
            )

    def test_parse_data_url(self):
        """Test parsing data URLs."""
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": MagicMock(),
                "google.genai.types": MagicMock(),
                "google.genai.errors": MagicMock(),
            },
        ):
            from crewai.llms.providers.gemini.completion import GeminiCompletion

            # Create a mock instance
            completion = MagicMock(spec=GeminiCompletion)
            completion._parse_data_url = GeminiCompletion._parse_data_url.__get__(
                completion, GeminiCompletion
            )

            # Create a test data URL
            test_data = b"test image data"
            base64_data = base64.b64encode(test_data).decode("utf-8")
            data_url = f"data:image/png;base64,{base64_data}"

            # Mock types.Part.from_bytes
            mock_part = MagicMock()
            with patch(
                "crewai.llms.providers.gemini.completion.types.Part.from_bytes",
                return_value=mock_part,
            ) as mock_from_bytes:
                result = completion._parse_data_url(data_url)

            # Verify from_bytes was called with correct arguments
            mock_from_bytes.assert_called_once()
            call_args = mock_from_bytes.call_args
            assert call_args.kwargs["mime_type"] == "image/png"
            assert call_args.kwargs["data"] == test_data

    def test_fetch_image_from_url(self):
        """Test fetching images from HTTP URLs."""
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": MagicMock(),
                "google.genai.types": MagicMock(),
                "google.genai.errors": MagicMock(),
            },
        ):
            from crewai.llms.providers.gemini.completion import GeminiCompletion

            # Create a mock instance
            completion = MagicMock(spec=GeminiCompletion)
            completion._fetch_image_from_url = (
                GeminiCompletion._fetch_image_from_url.__get__(
                    completion, GeminiCompletion
                )
            )

            # Mock urllib.request.urlopen
            mock_response = MagicMock()
            mock_response.read.return_value = b"fake image data"
            mock_response.headers.get.return_value = "image/jpeg"
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)

            mock_part = MagicMock()
            with (
                patch("urllib.request.urlopen", return_value=mock_response),
                patch(
                    "crewai.llms.providers.gemini.completion.types.Part.from_bytes",
                    return_value=mock_part,
                ) as mock_from_bytes,
            ):
                result = completion._fetch_image_from_url("https://example.com/image.jpg")

            # Verify from_bytes was called with correct arguments
            mock_from_bytes.assert_called_once()
            call_args = mock_from_bytes.call_args
            assert call_args.kwargs["mime_type"] == "image/jpeg"
            assert call_args.kwargs["data"] == b"fake image data"

    def test_read_local_image(self):
        """Test reading local image files."""
        import tempfile
        import os

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": MagicMock(),
                "google.genai.types": MagicMock(),
                "google.genai.errors": MagicMock(),
            },
        ):
            from crewai.llms.providers.gemini.completion import GeminiCompletion

            # Create a mock instance
            completion = MagicMock(spec=GeminiCompletion)
            completion._read_local_image = GeminiCompletion._read_local_image.__get__(
                completion, GeminiCompletion
            )

            # Create a temporary test file
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as temp_file:
                temp_file.write(b"fake png data")
                temp_path = temp_file.name

            try:
                mock_part = MagicMock()
                with patch(
                    "crewai.llms.providers.gemini.completion.types.Part.from_bytes",
                    return_value=mock_part,
                ) as mock_from_bytes:
                    result = completion._read_local_image(temp_path)

                # Verify from_bytes was called with correct arguments
                mock_from_bytes.assert_called_once()
                call_args = mock_from_bytes.call_args
                assert call_args.kwargs["mime_type"] == "image/png"
                assert call_args.kwargs["data"] == b"fake png data"
            finally:
                os.unlink(temp_path)

    def test_read_local_image_file_not_found(self):
        """Test reading non-existent local image file."""
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.genai": MagicMock(),
                "google.genai.types": MagicMock(),
                "google.genai.errors": MagicMock(),
            },
        ):
            from crewai.llms.providers.gemini.completion import GeminiCompletion

            # Create a mock instance
            completion = MagicMock(spec=GeminiCompletion)
            completion._read_local_image = GeminiCompletion._read_local_image.__get__(
                completion, GeminiCompletion
            )

            result = completion._read_local_image("/nonexistent/path/image.png")

            # Should return None for non-existent file
            assert result is None


class TestAddImageToolOutput:
    """Tests for AddImageTool output format."""

    def test_add_image_tool_returns_correct_format(self):
        """Test that AddImageTool returns the correct multimodal format."""
        from crewai.tools.agent_tools.add_image_tool import AddImageTool

        tool = AddImageTool()
        result = tool._run(
            image_url="https://example.com/image.jpg",
            action="Analyze this image",
        )

        # Verify the result format
        assert isinstance(result, dict)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2

        # Check text part
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Analyze this image"

        # Check image_url part
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"
