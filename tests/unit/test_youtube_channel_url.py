"""Tests for YoutubeChannelSearchTool URL normalization (issue #7404)."""

from unittest.mock import patch, MagicMock

import pytest

from crewai_tools.tools.youtube_channel_search_tool.youtube_channel_search_tool import (
    YoutubeChannelSearchTool,
)


class TestYoutubeChannelHandleNormalization:
    """The tool must accept @handles, bare names, and full URLs."""

    def test_add_with_handle(self):
        """@handle should be converted to full YouTube URL."""
        tool = YoutubeChannelSearchTool.__new__(YoutubeChannelSearchTool)
        # Call the add method and verify the URL passed to super().add
        with patch("crewai_tools.tools.youtube_channel_search_tool.youtube_channel_search_tool.RagTool.add") as mock_add:
            tool.add("@testchannel")
            mock_add.assert_called_once()
            url_arg = mock_add.call_args[0][0]
            assert url_arg == "https://www.youtube.com/@testchannel"

    def test_add_with_bare_name(self):
        """Bare channel name should be converted to full YouTube URL."""
        tool = YoutubeChannelSearchTool.__new__(YoutubeChannelSearchTool)
        with patch("crewai_tools.tools.youtube_channel_search_tool.youtube_channel_search_tool.RagTool.add") as mock_add:
            tool.add("testchannel")
            mock_add.assert_called_once()
            url_arg = mock_add.call_args[0][0]
            assert url_arg == "https://www.youtube.com/@testchannel"

    def test_add_with_full_url(self):
        """Full YouTube URL should be passed through unchanged."""
        tool = YoutubeChannelSearchTool.__new__(YoutubeChannelSearchTool)
        with patch("crewai_tools.tools.youtube_channel_search_tool.youtube_channel_search_tool.RagTool.add") as mock_add:
            tool.add("https://www.youtube.com/@testchannel")
            mock_add.assert_called_once()
            url_arg = mock_add.call_args[0][0]
            assert url_arg == "https://www.youtube.com/@testchannel"

    def test_add_with_full_url_no_at(self):
        """Full URL without @ should be passed through unchanged."""
        tool = YoutubeChannelSearchTool.__new__(YoutubeChannelSearchTool)
        with patch("crewai_tools.tools.youtube_channel_search_tool.youtube_channel_search_tool.RagTool.add") as mock_add:
            tool.add("https://www.youtube.com/channel/UC123456")
            mock_add.assert_called_once()
            url_arg = mock_add.call_args[0][0]
            assert url_arg == "https://www.youtube.com/channel/UC123456"
