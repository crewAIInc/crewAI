import json
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.youtube_channel_loader import (
    YoutubeChannelLoader,
    _normalize_channel_url,
)
from crewai_tools.rag.source_content import SourceContent


def make_lockup_item(video_id: str, title: str) -> dict[str, Any]:
    """Helper to build a modern YouTube lockupViewModel item."""
    return {
        "richItemRenderer": {
            "content": {
                "lockupViewModel": {
                    "contentId": video_id,
                    "metadata": {
                        "lockupMetadataViewModel": {
                            "title": {"content": title}
                        }
                    },
                }
            }
        }
    }


def make_continuation_item(token: str) -> dict[str, Any]:
    """Helper to build a continuationItemRenderer item."""
    return {
        "continuationItemRenderer": {
            "continuationEndpoint": {
                "continuationCommand": {
                    "token": token
                }
            }
        }
    }


def make_init_data(
    contents: list[Any], header: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Helper to build channel initial_data browse response."""
    res: dict[str, Any] = {
        "contents": {
            "twoColumnBrowseResultsRenderer": {
                "tabs": [
                    {
                        "tabRenderer": {
                            "title": "Videos",
                            "selected": True,
                            "content": {
                                "richGridRenderer": {
                                    "contents": contents
                                }
                            },
                        }
                    }
                ]
            }
        }
    }
    if header:
        res["header"] = header
    return res


def make_fireship_header(count_text: str = "841 videos") -> dict[str, Any]:
    """Helper to build pageHeaderViewModel with video count metadata."""
    return {
        "pageHeaderRenderer": {
            "content": {
                "pageHeaderViewModel": {
                    "metadata": {
                        "contentMetadataViewModel": {
                            "metadataRows": [
                                {
                                    "metadataParts": [
                                        {"text": {"content": count_text}}
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        }
    }


def make_continuation_response(items: list[Any]) -> str:
    """Helper to build continuation JSON response."""
    return json.dumps({
        "onResponseReceivedActions": [
            {
                "appendContinuationItemsAction": {
                    "continuationItems": items
                }
            }
        ]
    })


class TestYoutubeChannelLoader:
    def test_create_channel_internal_adaptation(self) -> None:
        """Test that _create_channel adapts @handle URLs without mutating pytube.extract."""
        from pytube import extract  # type: ignore[import-untyped]
        from pytube.exceptions import RegexMatchError  # type: ignore[import-untyped]

        ch = YoutubeChannelLoader._create_channel("https://www.youtube.com/@crewAI")
        assert ch.channel_uri == "/@crewAI"
        assert ch.channel_url == "https://www.youtube.com/@crewAI"
        assert ch.videos_url == "https://www.youtube.com/@crewAI/videos"

        # Verify that pytube.extract.channel_name was NOT globally mutated
        with pytest.raises(RegexMatchError):
            extract.channel_name("https://www.youtube.com/@crewAI")

    def test_normalize_channel_url(self) -> None:
        """Test URL normalization and percent-encoding for handles and URLs."""
        assert _normalize_channel_url("@crewAI") == "https://www.youtube.com/@crewAI"
        assert _normalize_channel_url("crewAI") == "https://www.youtube.com/@crewAI"
        assert _normalize_channel_url("@日本語") == "https://www.youtube.com/@%E6%97%A5%E6%9C%AC%E8%AA%9E"
        assert _normalize_channel_url("日本語") == "https://www.youtube.com/@%E6%97%A5%E6%9C%AC%E8%AA%9E"
        assert (
            _normalize_channel_url("https://www.youtube.com/@%E6%97%A5%E6%9C%AC%E8%AA%9E")
            == "https://www.youtube.com/@%E6%97%A5%E6%9C%AC%E8%AA%9E"
        )
        assert _normalize_channel_url("www.youtube.com/@crewAI") == "https://www.youtube.com/@crewAI"
        assert (
            _normalize_channel_url("https://www.youtube.com/channel/UC123")
            == "https://www.youtube.com/channel/UC123"
        )
        # Verify query and fragment are preserved on existing URLs
        assert (
            _normalize_channel_url("https://www.youtube.com/channel/UC123?feature=shared#section")
            == "https://www.youtube.com/channel/UC123?feature=shared#section"
        )
        # Verify case-insensitive scheme and hostname handling
        assert (
            _normalize_channel_url("HTTPS://WWW.YOUTUBE.COM/@crewAI")
            == "https://www.youtube.com/@crewAI"
        )
        assert (
            _normalize_channel_url("WWW.YOUTUBE.COM/@crewAI")
            == "https://www.youtube.com/@crewAI"
        )
        assert (
            _normalize_channel_url("HTTP://YOUTUBE.COM/CHANNEL/UC123?feature=shared#section")
            == "https://youtube.com/CHANNEL/UC123?feature=shared#section"
        )

    @patch("pytube.Channel")
    def test_load_with_at_handle(self, mock_channel_cls: Any) -> None:
        """Test loading a channel with @handle format."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "CrewAI"
        mock_channel.channel_id = "UC123456"
        mock_channel.video_urls = [
            "https://www.youtube.com/watch?v=12345678901",
            "https://www.youtube.com/watch?v=12345678902",
        ]
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        result = loader.load(SourceContent("@crewAI"))

        assert isinstance(result, LoaderResult)
        assert result.metadata["source"] == "@crewAI"
        assert result.metadata["data_type"] == "youtube_channel"
        assert result.metadata["channel_name"] == "CrewAI"
        assert result.metadata["channel_id"] == "UC123456"
        assert result.metadata["num_videos_loaded"] == 2
        assert result.metadata["total_videos"] == 2
        assert "YouTube Channel: CrewAI" in result.content
        assert "Channel ID: UC123456" in result.content
        assert "Total Videos: 2" in result.content
        assert "Videos Loaded: 2" in result.content

    @patch("pytube.Channel")
    def test_load_with_bare_handle(self, mock_channel_cls: Any) -> None:
        """Test loading a channel with bare handle format (no @ prefix)."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "CrewAI"
        mock_channel.channel_id = "UC123456"
        mock_channel.video_urls = ["https://www.youtube.com/watch?v=12345678901"]
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        result = loader.load(SourceContent("crewAI"))

        assert isinstance(result, LoaderResult)
        assert result.metadata["source"] == "crewAI"
        assert result.metadata["channel_name"] == "CrewAI"
        assert result.metadata["num_videos_loaded"] == 1

    @patch("pytube.Channel")
    def test_load_with_multilingual_handle(self, mock_channel_cls: Any) -> None:
        """Test loading a channel with non-ASCII characters in handle."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "日本語チャンネル"
        mock_channel.channel_id = "UCjapanese123"
        mock_channel.video_urls = ["https://www.youtube.com/watch?v=12345678901"]
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        result = loader.load(SourceContent("@日本語"))

        assert isinstance(result, LoaderResult)
        assert result.metadata["source"] == "@日本語"
        assert result.metadata["channel_name"] == "日本語チャンネル"
        assert result.metadata["num_videos_loaded"] == 1
        assert "YouTube Channel: 日本語チャンネル" in result.content

    @patch("pytube.Channel")
    def test_load_with_full_handle_url(self, mock_channel_cls: Any) -> None:
        """Test loading a channel with full https://www.youtube.com/@handle URL."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "CrewAI"
        mock_channel.channel_id = "UC123456"
        mock_channel.video_urls = ["https://www.youtube.com/watch?v=12345678901"]
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        result = loader.load(SourceContent("https://www.youtube.com/@crewAI"))

        assert isinstance(result, LoaderResult)
        assert result.metadata["source"] == "https://www.youtube.com/@crewAI"
        assert result.metadata["num_videos_loaded"] == 1

    @patch("pytube.Channel")
    def test_load_with_channel_id_url(self, mock_channel_cls: Any) -> None:
        """Test loading a channel with traditional channel ID URL."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "CrewAI"
        mock_channel.channel_id = "UC_x5XG1OV2P6uZZ5FSM9Ttw"
        mock_channel.video_urls = ["https://www.youtube.com/watch?v=12345678901"]
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        result = loader.load(
            SourceContent("https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw")
        )

        assert isinstance(result, LoaderResult)
        assert result.metadata["channel_name"] == "CrewAI"
        assert result.metadata["num_videos_loaded"] == 1

    @patch("pytube.Channel")
    def test_load_with_c_url(self, mock_channel_cls: Any) -> None:
        """Test loading a channel with legacy /c/ URL."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "CrewAI"
        mock_channel.channel_id = "UC123456"
        mock_channel.video_urls = ["https://www.youtube.com/watch?v=12345678901"]
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        result = loader.load(SourceContent("https://www.youtube.com/c/crewAI"))

        assert isinstance(result, LoaderResult)
        assert result.metadata["channel_name"] == "CrewAI"
        assert result.metadata["num_videos_loaded"] == 1

    @patch("pytube.Channel")
    def test_load_with_schemeless_url(self, mock_channel_cls: Any) -> None:
        """Test loading a channel URL without http/https scheme."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "CrewAI"
        mock_channel.channel_id = "UC123456"
        mock_channel.video_urls = ["https://www.youtube.com/watch?v=12345678901"]
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        result = loader.load(SourceContent("www.youtube.com/@crewAI"))

        assert isinstance(result, LoaderResult)
        assert result.metadata["channel_name"] == "CrewAI"
        assert result.metadata["num_videos_loaded"] == 1

    @patch("pytube.Channel")
    def test_load_preserves_raw_source_and_doc_id(self, mock_channel_cls: Any) -> None:
        """Test that existing URLs with query and fragment preserve source identifier and doc_id."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "QueryChannel"
        mock_channel.channel_id = "UCquery123"
        mock_channel.video_urls = ["https://www.youtube.com/watch?v=12345678901"]
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        raw_url = "https://www.youtube.com/channel/UCquery123?feature=shared#section"
        loader = YoutubeChannelLoader()
        result = loader.load(SourceContent(raw_url), max_videos=1)

        assert result.source == raw_url
        assert result.metadata["source"] == raw_url
        assert result.doc_id == loader.generate_doc_id(
            source_ref=raw_url, content=result.content
        )

    @patch("pytube.Channel")
    def test_load_non_empty_channel_youtube_details_success(
        self, mock_channel_cls: Any
    ) -> None:
        """Test that modern richGridRenderer structure extracts videos with YouTube details."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "Fireship"
        mock_channel.channel_id = "UCsBjURrPoezykLs9EqgamOA"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data(
            [
                make_lockup_item("7xTGNNLPyMI", "100+ Computer Science Concepts Explained"),
                make_lockup_item("DC4brKmA74w", "React in 100 Seconds"),
            ],
            header=make_fireship_header("841 videos"),
        )

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            def make_mock_yt(url: str) -> MagicMock:
                """Create a mock YouTube video instance matching the given URL."""
                yt = MagicMock()
                if "7xTGNNLPyMI" in url:
                    yt.title = "Detailed: 100+ Computer Science Concepts"
                    yt.description = "Full description for CS concepts"
                else:
                    yt.title = "Detailed: React in 100 Seconds"
                    yt.description = "Full description for React"
                return yt

            mock_youtube_cls.side_effect = make_mock_yt

            mock_entry = MagicMock()
            mock_entry.text = "Hello world"
            mock_transcript_obj = MagicMock()
            mock_transcript_obj.fetch.return_value = [mock_entry]
            mock_transcript_list = MagicMock()
            mock_transcript_list.find_transcript.return_value = mock_transcript_obj
            mock_transcript_api.return_value.list.return_value = mock_transcript_list

            loader = YoutubeChannelLoader()
            result = loader.load(SourceContent("@Fireship"), max_videos=10)

        assert result.metadata["num_videos_loaded"] == 2
        assert result.metadata["total_videos"] == 841
        assert "Total Videos: 841" in result.content
        assert "Videos Loaded: 2" in result.content
        assert "1. Detailed: 100+ Computer Science Concepts" in result.content
        assert "URL: https://www.youtube.com/watch?v=7xTGNNLPyMI" in result.content
        assert "Description: Full description for CS concepts..." in result.content
        assert "Transcript Preview: Hello world..." in result.content
        assert "2. Detailed: React in 100 Seconds" in result.content
        assert "URL: https://www.youtube.com/watch?v=DC4brKmA74w" in result.content

    @patch("pytube.Channel")
    def test_load_non_empty_channel_youtube_details_fallback(
        self, mock_channel_cls: Any
    ) -> None:
        """Test fallback to lockupViewModel title when pytube.YouTube details request fails."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "Fireship"
        mock_channel.channel_id = "UCsBjURrPoezykLs9EqgamOA"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("7xTGNNLPyMI", "100+ Computer Science Concepts Explained"),
            make_lockup_item("DC4brKmA74w", "React in 100 Seconds"),
        ])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.YouTube", side_effect=Exception("Innertube HTTP 400")),
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            result = loader.load(SourceContent("@Fireship"), max_videos=10)

        assert result.metadata["num_videos_loaded"] == 2
        assert "1. 100+ Computer Science Concepts Explained" in result.content
        assert "URL: https://www.youtube.com/watch?v=7xTGNNLPyMI" in result.content
        assert "Description: No description..." in result.content
        assert "2. React in 100 Seconds" in result.content
        assert "URL: https://www.youtube.com/watch?v=DC4brKmA74w" in result.content

    @patch("pytube.Channel")
    def test_load_modern_structure_pagination(self, mock_channel_cls: Any) -> None:
        """Test pagination using continuationItemRenderer when max_videos exceeds first screen."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "Fireship"
        mock_channel.channel_id = "UCsBjURrPoezykLs9EqgamOA"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel._build_continuation_url.return_value = (
            "https://www.youtube.com/youtubei/v1/browse?key=dummy_key",
            {"header": "val"},
            {"continuation": "token_page_2"},
        )
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data(
            [
                make_lockup_item("video000001", "Video 1"),
                make_lockup_item("video000002", "Video 2"),
                make_continuation_item("token_page_2"),
            ],
            header=make_fireship_header("841 videos"),
        )

        mock_page2_resp = make_continuation_response([
            make_lockup_item("video000003", "Video 3"),
            make_lockup_item("video000004", "Video 4"),
        ])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.request.post", return_value=mock_page2_resp) as mock_post,
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = None
            mock_yt.description = None
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            result = loader.load(SourceContent("@Fireship"), max_videos=3)

        mock_post.assert_called_once()
        assert result.metadata["num_videos_loaded"] == 3
        assert result.metadata["total_videos"] == 841
        assert "Total Videos: 841" in result.content
        assert "Videos Loaded: 3" in result.content
        assert "1. Video 1" in result.content
        assert "2. Video 2" in result.content
        assert "3. Video 3" in result.content
        assert "4. Video 4" not in result.content

    @patch("pytube.Channel")
    def test_load_modern_structure_no_header_count_with_continuation(
        self, mock_channel_cls: Any
    ) -> None:
        """Test that first page count is not falsely reported as total when continuation exists."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "PartialChannel"
        mock_channel.channel_id = "UCpartial123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("video000001", "Video 1"),
            make_continuation_item("token_page_2"),
        ])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = None
            mock_yt.description = None
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            result = loader.load(SourceContent("@PartialChannel"), max_videos=1)

        assert result.metadata["num_videos_loaded"] == 1
        assert result.metadata["total_videos"] is None
        assert "Total Videos: Unknown" in result.content
        assert "Videos Loaded: 1" in result.content

    @patch("pytube.Channel")
    def test_load_modern_structure_pagination_list_response(
        self, mock_channel_cls: Any
    ) -> None:
        """Test continuation pagination when response is returned as a JSON list."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "ListResponseChannel"
        mock_channel.channel_id = "UClist123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel._build_continuation_url.return_value = (
            "https://www.youtube.com/youtubei/v1/browse?key=dummy_key",
            {"header": "val"},
            {"continuation": "token_page_2"},
        )
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("video000001", "Video 1"),
            make_continuation_item("token_page_2"),
        ])

        mock_page2_resp = json.dumps([
            {"other": "data"},
            {
                "response": {
                    "onResponseReceivedActions": [
                        {
                            "appendContinuationItemsAction": {
                                "continuationItems": [
                                    make_lockup_item("video000002", "Video 2")
                                ]
                            }
                        }
                    ]
                }
            },
        ])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.request.post", return_value=mock_page2_resp) as mock_post,
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = None
            mock_yt.description = None
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            result = loader.load(SourceContent("@ListResponseChannel"), max_videos=2)

        mock_post.assert_called_once()
        assert result.metadata["num_videos_loaded"] == 2
        assert "1. Video 1" in result.content
        assert "2. Video 2" in result.content

    @patch("pytube.Channel")
    def test_load_modern_structure_unrecognized_continuation_response_raises(
        self, mock_channel_cls: Any
    ) -> None:
        """Test that unrecognized continuation responses raise ValueError instead of returning partial."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "UnrecognizedChannel"
        mock_channel.channel_id = "UCunrec123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel._build_continuation_url.return_value = (
            "https://www.youtube.com/youtubei/v1/browse?key=dummy_key",
            {"header": "val"},
            {"continuation": "token_page_2"},
        )
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("video000001", "Video 1"),
            make_continuation_item("token_page_2"),
        ])

        mock_page2_resp = json.dumps({"responseContext": {}})

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.request.post", return_value=mock_page2_resp),
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = None
            mock_yt.description = None
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            with pytest.raises(
                ValueError, match="Unrecognized YouTube continuation response"
            ):
                loader.load(SourceContent("@UnrecognizedChannel"), max_videos=2)

    @patch("pytube.Channel")
    def test_load_modern_structure_explicit_empty_terminal_continuation(
        self, mock_channel_cls: Any
    ) -> None:
        """Test that explicit empty continuation items terminates pagination cleanly as channel end."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "TerminalEmptyChannel"
        mock_channel.channel_id = "UCterm123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel._build_continuation_url.return_value = (
            "https://www.youtube.com/youtubei/v1/browse?key=dummy_key",
            {"header": "val"},
            {"continuation": "token_page_2"},
        )
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("video000001", "Video 1"),
            make_continuation_item("token_page_2"),
        ])

        mock_page2_resp = make_continuation_response([])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.request.post", return_value=mock_page2_resp) as mock_post,
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = None
            mock_yt.description = None
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            result = loader.load(SourceContent("@TerminalEmptyChannel"), max_videos=10)

        mock_post.assert_called_once()
        assert result.metadata["num_videos_loaded"] == 1
        assert result.metadata["total_videos"] == 1
        assert "Total Videos: 1" in result.content
        assert "Videos Loaded: 1" in result.content
        assert "1. Video 1" in result.content

    @pytest.mark.parametrize(
        "payload,expected_err",
        [
            (
                {"onResponseReceivedActions": []},
                "Unrecognized YouTube continuation response format",
            ),
            (
                {
                    "onResponseReceivedActions": [
                        {"appendContinuationItemsAction": {}}
                    ]
                },
                "Missing continuationItems in YouTube continuation action",
            ),
            (
                {
                    "onResponseReceivedActions": [
                        {"appendContinuationItemsAction": {"continuationItems": "invalid"}}
                    ]
                },
                "continuationItems must be a list in YouTube continuation action",
            ),
            (
                {
                    "onResponseReceivedActions": [
                        {"appendContinuationItemsAction": {"continuationItems": None}}
                    ]
                },
                "continuationItems must be a list in YouTube continuation action",
            ),
        ],
    )
    @patch("pytube.Channel")
    def test_load_modern_structure_invalid_continuation_payloads_raise(
        self, mock_channel_cls: Any, payload: Any, expected_err: str
    ) -> None:
        """Test that missing or invalid continuation actions/items raise ValueError."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "InvalidPayloadChannel"
        mock_channel.channel_id = "UCinvalid123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel._build_continuation_url.return_value = (
            "https://www.youtube.com/youtubei/v1/browse?key=dummy_key",
            {"header": "val"},
            {"continuation": "token_page_2"},
        )
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("video000001", "Video 1"),
            make_continuation_item("token_page_2"),
        ])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.request.post", return_value=json.dumps(payload)),
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = None
            mock_yt.description = None
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            with pytest.raises(ValueError, match=expected_err):
                loader.load(SourceContent("@InvalidPayloadChannel"), max_videos=5)

    @patch("pytube.Channel")
    def test_load_modern_structure_pagination_skips_duplicate_page_and_continues(
        self, mock_channel_cls: Any
    ) -> None:
        """Test that pagination continues when a page contains duplicates but provides a next token."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "DupPageChannel"
        mock_channel.channel_id = "UCdup123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel._build_continuation_url.side_effect = lambda tok: (
            f"https://www.youtube.com/youtubei/v1/browse?token={tok}",
            {"header": "val"},
            {"continuation": tok},
        )
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("video000001", "Video 1"),
            make_continuation_item("token_page_2"),
        ])

        # Page 2 returns duplicate video000001, but provides token_page_3
        mock_page2_resp = make_continuation_response([
            make_lockup_item("video000001", "Video 1 duplicate"),
            make_continuation_item("token_page_3"),
        ])

        # Page 3 returns fresh video000002
        mock_page3_resp = make_continuation_response([
            make_lockup_item("video000002", "Video 2"),
        ])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.request.post", side_effect=[mock_page2_resp, mock_page3_resp]) as mock_post,
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = None
            mock_yt.description = None
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            result = loader.load(SourceContent("@DupPageChannel"), max_videos=2)

        assert mock_post.call_count == 2
        assert result.metadata["num_videos_loaded"] == 2
        assert "1. Video 1" in result.content
        assert "2. Video 2" in result.content

    @patch("pytube.Channel")
    def test_load_modern_structure_pagination_circular_token_terminates(
        self, mock_channel_cls: Any
    ) -> None:
        """Test that pagination explicitly raises ValueError when next token is circular / repeated."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "CircularChannel"
        mock_channel.channel_id = "UCcirc123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel._build_continuation_url.return_value = (
            "https://www.youtube.com/youtubei/v1/browse?key=dummy_key",
            {"header": "val"},
            {"continuation": "token_repeat"},
        )
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("video000001", "Video 1"),
            make_continuation_item("token_repeat"),
        ])

        # Page 2 returns video000002, but repeats token_repeat
        mock_page2_resp = make_continuation_response([
            make_lockup_item("video000002", "Video 2"),
            make_continuation_item("token_repeat"),
        ])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch("pytube.request.post", return_value=mock_page2_resp) as mock_post,
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = None
            mock_yt.description = None
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            # Request 10 videos; circular token should raise ValueError explicitly
            with pytest.raises(
                ValueError,
                match="Detected circular continuation token: token_repeat",
            ):
                loader.load(SourceContent("@CircularChannel"), max_videos=10)

        assert mock_post.call_count == 1

    @patch("pytube.Channel")
    def test_load_modern_structure_pagination_request_failure_preserves_partial(
        self, mock_channel_cls: Any
    ) -> None:
        """Test that continuation request or JSON parse failures stop pagination and preserve already collected videos."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "TransientFailChannel"
        mock_channel.channel_id = "UCtrans123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel._build_continuation_url.return_value = (
            "https://www.youtube.com/youtubei/v1/browse?key=dummy_key",
            {"header": "val"},
            {"continuation": "token_fail"},
        )
        mock_channel_cls.return_value = mock_channel

        mock_init_data = make_init_data([
            make_lockup_item("video000001", "Video 1"),
            make_continuation_item("token_fail"),
        ])

        with (
            patch("pytube.extract.initial_data", return_value=mock_init_data),
            patch(
                "pytube.request.post",
                side_effect=RuntimeError("Connection reset by peer"),
            ) as mock_post,
            patch("pytube.YouTube") as mock_youtube_cls,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_transcript_api,
        ):
            mock_yt = MagicMock()
            mock_yt.title = "Video 1"
            mock_yt.description = "Desc 1"
            mock_youtube_cls.return_value = mock_yt
            mock_transcript_api.return_value.list.side_effect = Exception("No transcript")

            loader = YoutubeChannelLoader()
            result = loader.load(SourceContent("@TransientFailChannel"), max_videos=10)

        assert mock_post.call_count == 1
        assert result.metadata["num_videos_loaded"] == 1
        assert result.metadata["total_videos"] is None
        assert "Total Videos: Unknown" in result.content
        assert "1. Video 1" in result.content

    @patch("pytube.Channel")
    def test_load_video_list_error_propagates(self, mock_channel_cls: Any) -> None:
        """Test that network or parsing exceptions during video list loading are not swallowed."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "ErrorChannel"
        mock_channel.channel_id = "UCerror123"
        type(mock_channel).video_urls = PropertyMock(
            side_effect=RuntimeError("Network connection reset")
        )
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        with pytest.raises(
            ValueError, match="Unable to load YouTube channel.*Network connection reset"
        ):
            loader.load(SourceContent("@ErrorChannel"))

    def test_load_invalid_url_raises_value_error(self) -> None:
        """Test that invalid URLs raise ValueError."""
        loader = YoutubeChannelLoader()

        with pytest.raises(ValueError, match="Invalid YouTube channel URL"):
            loader.load(SourceContent("https://example.com/not-youtube"))

    @patch("pytube.Channel")
    def test_load_channel_failure_raises_value_error(
        self, mock_channel_cls: Any
    ) -> None:
        """Test that channel initialization failures raise ValueError with informative message."""
        mock_channel_cls.side_effect = Exception("Channel not found")

        loader = YoutubeChannelLoader()
        with pytest.raises(
            ValueError, match="Unable to load YouTube channel.*Channel not found"
        ):
            loader.load(SourceContent("@NonExistentChannel"))

    @patch("crewai_tools.rag.loaders.youtube_channel_loader.YoutubeChannelLoader._create_channel")
    def test_load_attacker_host_rejected(self, mock_create: Any) -> None:
        """Test that non-YouTube hostnames with youtube.com in path are rejected before channel creation."""
        loader = YoutubeChannelLoader()
        with pytest.raises(ValueError, match="Invalid YouTube channel URL"):
            loader.load(SourceContent("https://attacker.example/youtube.com/channel/UC123"))
        mock_create.assert_not_called()

    @patch("pytube.Channel")
    def test_load_with_uppercase_channel_url_and_query_fragment(
        self, mock_channel_cls: Any
    ) -> None:
        """Test loading a channel with uppercase route and query/fragment parameters."""
        mock_channel = MagicMock()
        mock_channel.channel_name = "UC123"
        mock_channel.channel_id = "UC123"
        mock_channel.video_urls = []
        mock_channel.html = "<html>mock</html>"
        mock_channel_cls.return_value = mock_channel

        loader = YoutubeChannelLoader()
        with (
            patch("pytube.extract.initial_data", return_value={}),
            patch("pytube.YouTube"),
            patch("youtube_transcript_api.YouTubeTranscriptApi"),
        ):
            result = loader.load(
                SourceContent("HTTP://YOUTUBE.COM/CHANNEL/UC123?feature=shared#section")
            )
            assert result.source == "HTTP://YOUTUBE.COM/CHANNEL/UC123?feature=shared#section"
            assert mock_channel_cls.call_count == 1
