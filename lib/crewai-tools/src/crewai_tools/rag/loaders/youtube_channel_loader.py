"""YouTube channel loader for extracting content from YouTube channels."""

import json
import logging
import re
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


logger = logging.getLogger(__name__)


def _normalize_channel_url(raw_input: str) -> str:
    """Normalize a channel handle or URL into a valid YouTube channel request URL with proper percent-encoding."""
    cleaned = raw_input.strip()
    if cleaned.startswith("@"):
        path = f"/{cleaned}"
        netloc = "www.youtube.com"
        scheme = "https"
        query = ""
        fragment = ""
    else:
        lower_cleaned = cleaned.lower()
        if not any(
            p in lower_cleaned
            for p in [
                "youtube.com/",
                "youtu.be/",
                "http://",
                "https://",
            ]
        ):
            path = f"/@{cleaned}"
            netloc = "www.youtube.com"
            scheme = "https"
            query = ""
            fragment = ""
        else:
            if not lower_cleaned.startswith(("http://", "https://")):
                cleaned = f"https://{cleaned}"
            parts = urlsplit(cleaned)
            scheme = "https"
            netloc = (parts.netloc or "www.youtube.com").lower()
            path = parts.path
            query = parts.query
            fragment = parts.fragment

    encoded_path = quote(unquote(path), safe="/:@%~_.-")
    return str(urlunsplit((scheme, netloc, encoded_path, query, fragment)))


class YoutubeChannelLoader(BaseLoader):
    """Loader for YouTube channels."""

    @classmethod
    def _create_channel(cls, channel_url: str) -> Any:
        """Create a pytube Channel instance with internal compatibility adaptation for @handle URLs."""
        from pytube import Channel, Playlist  # type: ignore[import-untyped]
        from pytube.exceptions import RegexMatchError  # type: ignore[import-untyped]

        try:
            return Channel(channel_url)
        except RegexMatchError:
            match = re.search(r"(/@([^/?#]+))", channel_url)
            if not match:
                raise
            channel = Channel.__new__(Channel)
            Playlist.__init__(channel, channel_url)
            channel.channel_uri = str(match.group(1))
            channel.channel_url = f"https://www.youtube.com{channel.channel_uri}"
            channel.videos_url = channel.channel_url + "/videos"
            channel.playlists_url = channel.channel_url + "/playlists"
            channel.community_url = channel.channel_url + "/community"
            channel.featured_channels_url = channel.channel_url + "/channels"
            channel.about_url = channel.channel_url + "/about"
            channel._playlists_html = None
            channel._community_html = None
            channel._featured_channels_html = None
            channel._about_html = None
            return channel

    def load(self, source: SourceContent, **kwargs: Any) -> LoaderResult:  # type: ignore[override]
        """Load and extract content from a YouTube channel.

        Args:
            source: The source content containing the YouTube channel URL

        Returns:
            LoaderResult with channel content

        Raises:
            ImportError: If required YouTube libraries aren't installed
            ValueError: If the URL is not a valid YouTube channel URL
        """
        try:
            import pytube  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "YouTube channel support requires pytube. Install with: uv add pytube"
            ) from e

        raw_source = source.source
        request_url = _normalize_channel_url(raw_source)
        parsed = urlsplit(request_url)
        allowed_hosts = {"www.youtube.com", "youtube.com", "m.youtube.com"}
        lower_path = parsed.path.lower()

        if parsed.netloc not in allowed_hosts or not any(
            lower_path.startswith(prefix)
            for prefix in [
                "/channel/",
                "/c/",
                "/@",
                "/user/",
            ]
        ):
            raise ValueError(f"Invalid YouTube channel URL: {raw_source}")

        metadata: dict[str, Any] = {
            "source": raw_source,
            "data_type": "youtube_channel",
        }

        try:
            channel = self._create_channel(request_url)

            metadata["channel_name"] = channel.channel_name
            metadata["channel_id"] = channel.channel_id

            max_videos = kwargs.get("max_videos", 10)
            video_items, total_videos = self._extract_channel_video_items(
                channel, max_videos=max_videos
            )
            metadata["num_videos_loaded"] = len(video_items)
            metadata["total_videos"] = total_videos

            total_str = str(total_videos) if total_videos is not None else "Unknown"
            content_parts = [
                f"YouTube Channel: {channel.channel_name}",
                f"Channel ID: {channel.channel_id}",
                f"Total Videos: {total_str}",
                f"Videos Loaded: {metadata['num_videos_loaded']}",
                "\n--- Video Summaries ---\n",
            ]

            try:
                from pytube import YouTube
                from youtube_transcript_api import YouTubeTranscriptApi

                for i, item in enumerate(video_items, 1):
                    video_url = item["url"]
                    video_id = item.get("id") or self._extract_video_id(video_url)
                    fallback_title = item.get("title")
                    try:
                        title = fallback_title or f"Video {i}"
                        description = "No description"
                        try:
                            yt = YouTube(video_url)
                            if yt.title:
                                title = yt.title
                            if yt.description:
                                description = yt.description[:200]
                        except Exception as e:
                            logger.debug(
                                "Failed to fetch YouTube video details for %s: %s",
                                video_url,
                                e,
                            )

                        content_parts.append(f"\n{i}. {title}")
                        content_parts.append(f"   URL: {video_url}")
                        content_parts.append(f"   Description: {description}...")

                        if video_id:
                            try:
                                api = YouTubeTranscriptApi()
                                transcript_list = api.list(video_id)

                                try:
                                    transcript = transcript_list.find_transcript(["en"])
                                except Exception:
                                    try:
                                        transcript = (
                                            transcript_list.find_generated_transcript(
                                                ["en"]
                                            )
                                        )
                                    except Exception:
                                        transcript = next(iter(transcript_list))

                                if transcript:
                                    transcript_data = transcript.fetch()
                                    text_parts = []
                                    char_count = 0
                                    for entry in transcript_data:
                                        text = (
                                            entry.text.strip()
                                            if hasattr(entry, "text")
                                            else ""
                                        )
                                        if text:
                                            text_parts.append(text)
                                            char_count += len(text)
                                            if char_count > 500:
                                                break

                                    if text_parts:
                                        preview = " ".join(text_parts)[:500]
                                        content_parts.append(
                                            f"   Transcript Preview: {preview}..."
                                        )
                            except Exception:
                                content_parts.append("   Transcript: Not available")

                    except Exception as e:
                        content_parts.append(f"\n{i}. Error loading video: {e!s}")

            except ImportError:
                for i, item in enumerate(video_items, 1):
                    content_parts.append(f"\n{i}. {item['url']}")

            content = "\n".join(content_parts)

        except Exception as e:
            raise ValueError(
                f"Unable to load YouTube channel {raw_source}: {e!s}"
            ) from e

        return LoaderResult(
            content=content,
            source=raw_source,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=raw_source, content=content),
        )

    @classmethod
    def _extract_channel_video_items(
        cls, channel: Any, max_videos: int = 10
    ) -> tuple[list[dict[str, str]], int | None]:
        """Extract video URLs, metadata, and total videos from a YouTube Channel.

        Supports both pytube's video_urls property and modern YouTube channel
        page structures (richGridRenderer, richItemRenderer, lockupViewModel)
        with continuation pagination.
        """
        video_items: list[dict[str, str]] = []
        seen_ids: set[str] = set()

        def _add_video(vid: str | None, title: str | None = None) -> bool:
            """Add a video item if valid and not already processed."""
            if vid and vid not in seen_ids and len(vid) == 11:
                seen_ids.add(vid)
                item: dict[str, str] = {
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "id": vid,
                }
                if title:
                    item["title"] = str(title)
                video_items.append(item)
                return True
            return False

        # 1. Try pytube's built-in video_urls first.
        # Note: Do not swallow exceptions so genuine network/read errors propagate.
        pytube_urls: list[str] = list(channel.video_urls)
        if pytube_urls:
            for url in pytube_urls:
                vid = cls._extract_video_id(url)
                _add_video(vid)
            total = len(video_items)
            return video_items[:max_videos], total

        # 2. Extract from channel.html using initial_data
        from pytube import extract

        init_data: dict[str, Any] = extract.initial_data(channel.html)
        total_videos_count = cls._extract_total_videos_count(init_data)

        continuation_token: str | None = None

        tabs = (
            init_data.get("contents", {})
            .get("twoColumnBrowseResultsRenderer", {})
            .get("tabs", [])
        )

        for tab in tabs:
            tr = tab.get("tabRenderer", {})
            content = tr.get("content", {})
            if "richGridRenderer" in content:
                for item in content["richGridRenderer"].get("contents", []):
                    parsed = cls._parse_video_renderer_item(item)
                    if parsed:
                        _add_video(parsed["id"], parsed.get("title"))
                    token = cls._extract_continuation_token(item)
                    if token:
                        continuation_token = token
            elif "sectionListRenderer" in content:
                for section in content["sectionListRenderer"].get("contents", []):
                    if "itemSectionRenderer" in section:
                        for sub_item in section["itemSectionRenderer"].get(
                            "contents", []
                        ):
                            if "gridRenderer" in sub_item:
                                for item in sub_item["gridRenderer"].get("items", []):
                                    parsed = cls._parse_video_renderer_item(item)
                                    if parsed:
                                        _add_video(parsed["id"], parsed.get("title"))
                                    token = cls._extract_continuation_token(item)
                                    if token:
                                        continuation_token = token
                            elif "playlistVideoListRenderer" in sub_item:
                                for item in sub_item["playlistVideoListRenderer"].get(
                                    "contents", []
                                ):
                                    parsed = cls._parse_video_renderer_item(item)
                                    if parsed:
                                        _add_video(parsed["id"], parsed.get("title"))
                                    token = cls._extract_continuation_token(item)
                                    if token:
                                        continuation_token = token
                            else:
                                parsed = cls._parse_video_renderer_item(sub_item)
                                if parsed:
                                    _add_video(parsed["id"], parsed.get("title"))
                                token = cls._extract_continuation_token(sub_item)
                                if token:
                                    continuation_token = token
                    else:
                        parsed = cls._parse_video_renderer_item(section)
                        if parsed:
                            _add_video(parsed["id"], parsed.get("title"))
                        token = cls._extract_continuation_token(section)
                        if token:
                            continuation_token = token

        # 3. Handle pagination if more videos are needed and continuation token exists
        seen_tokens: set[str] = set()

        while continuation_token and len(video_items) < max_videos:
            if continuation_token in seen_tokens:
                raise ValueError(
                    f"Detected circular continuation token: {continuation_token}"
                )

            seen_tokens.add(continuation_token)

            url, headers, data = cls._build_continuation_request(
                channel, continuation_token
            )
            from pytube import request

            try:
                resp_str = request.post(url, extra_headers=headers, data=data)
                resp_data: Any = json.loads(resp_str)
            except Exception as e:
                logger.debug("Failed to fetch YouTube continuation page: %s", e)
                break

            actions: list[dict[str, Any]] | None = None
            if isinstance(resp_data, dict):
                if "onResponseReceivedActions" in resp_data:
                    raw_actions = resp_data.get("onResponseReceivedActions")
                    if isinstance(raw_actions, list):
                        actions = raw_actions
            elif (
                isinstance(resp_data, list)
                and len(resp_data) > 1
                and isinstance(resp_data[1], dict)
            ):
                response_dict = resp_data[1].get("response")
                if (
                    isinstance(response_dict, dict)
                    and "onResponseReceivedActions" in response_dict
                ):
                    raw_actions = response_dict.get("onResponseReceivedActions")
                    if isinstance(raw_actions, list):
                        actions = raw_actions

            if actions is None or not actions:
                raise ValueError(
                    f"Unrecognized YouTube continuation response format: {resp_data!r}"
                )

            continuation_actions: list[dict[str, Any]] = []
            for a in actions:
                if not isinstance(a, dict):
                    continue
                for key in (
                    "appendContinuationItemsAction",
                    "reloadContinuationItemsCommand",
                ):
                    if key in a:
                        action_body = a[key]
                        if not isinstance(action_body, dict):
                            raise ValueError(
                                f"Invalid continuation action payload in YouTube response: {a!r}"
                            )
                        if "continuationItems" not in action_body:
                            raise ValueError(
                                f"Missing continuationItems in YouTube continuation action: {a!r}"
                            )
                        items_val = action_body["continuationItems"]
                        if not isinstance(items_val, list):
                            raise ValueError(
                                f"continuationItems must be a list in YouTube continuation action: {a!r}"
                            )
                        continuation_actions.append(action_body)

            if not continuation_actions:
                raise ValueError(
                    f"Unrecognized YouTube continuation response actions: {actions!r}"
                )

            next_token: str | None = None

            for action_body in continuation_actions:
                continuation_items = action_body["continuationItems"]
                for item in continuation_items:
                    if isinstance(item, dict):
                        parsed = cls._parse_video_renderer_item(item)
                        if parsed:
                            _add_video(parsed["id"], parsed.get("title"))
                        token = cls._extract_continuation_token(item)
                        if token:
                            next_token = token

            if next_token and next_token in seen_tokens:
                raise ValueError(f"Detected circular continuation token: {next_token}")

            continuation_token = next_token

        if total_videos_count is None:
            try:
                raw_len = channel.length
                if type(raw_len) is int:
                    total_videos_count = raw_len
            except Exception as e:
                logger.debug("Failed to extract channel length: %s", e)

        if total_videos_count is None:
            if not continuation_token:
                total_videos_count = len(video_items)
            else:
                total_videos_count = None

        return video_items[:max_videos], total_videos_count

    @classmethod
    def _extract_total_videos_count(cls, init_data: dict[str, Any]) -> int | None:
        """Extract total video count from channel initial_data if available."""
        header = init_data.get("header", {})

        def _parse_count(text: str) -> int | None:
            """Parse numeric video count from localized or abbreviated header text."""
            if not text:
                return None
            m = re.search(
                r"([\d,]+(?:\.\d+)?)\s*([kKmMbB])?\s*(?:video|videos|支影片|本の動画)",
                text,
                re.IGNORECASE,
            )
            if not m:
                return None
            num_str = m.group(1).replace(",", "")
            multiplier = 1
            suffix = m.group(2)
            if suffix:
                s = suffix.lower()
                if s == "k":
                    multiplier = 1_000
                elif s == "m":
                    multiplier = 1_000_000
                elif s == "b":
                    multiplier = 1_000_000_000
            try:
                return int(float(num_str) * multiplier)
            except ValueError:
                return None

        # 1. pageHeaderRenderer -> content -> pageHeaderViewModel
        vm = (
            header.get("pageHeaderRenderer", {})
            .get("content", {})
            .get("pageHeaderViewModel", {})
        )
        rows = (
            vm.get("metadata", {})
            .get("contentMetadataViewModel", {})
            .get("metadataRows", [])
        )
        for row in rows:
            for part in row.get("metadataParts", []):
                text_content = part.get("text", {}).get("content", "")
                count = _parse_count(text_content)
                if count is not None:
                    return count
                acc_label = part.get("accessibilityLabel", "")
                count = _parse_count(acc_label)
                if count is not None:
                    return count

        # 2. c4TabbedHeaderRenderer
        c4 = header.get("c4TabbedHeaderRenderer", {})
        vct = c4.get("videosCountText", {})
        if "runs" in vct:
            runs_text = "".join(str(r.get("text", "")) for r in vct.get("runs", []))
            count = _parse_count(runs_text)
            if count is not None:
                return count
        if "simpleText" in vct:
            count = _parse_count(str(vct.get("simpleText", "")))
            if count is not None:
                return count

        # 3. Fallback: recursively search header
        def _search_dict(d: Any) -> int | None:
            """Recursively search dictionary or list structures for video count text."""
            if isinstance(d, dict):
                for v in d.values():
                    res = _search_dict(v)
                    if res is not None:
                        return res
            elif isinstance(d, list):
                for v in d:
                    res = _search_dict(v)
                    if res is not None:
                        return res
            elif isinstance(d, str):
                return _parse_count(d)
            return None

        return _search_dict(header)

    @classmethod
    def _parse_video_renderer_item(cls, item: dict[str, Any]) -> dict[str, str] | None:
        """Parse a video renderer item into a dict with 'url', 'id', and optional 'title'."""
        vid: str | None = None
        title: str | None = None

        if "richItemRenderer" in item:
            inner = item["richItemRenderer"].get("content", {})
            if "lockupViewModel" in inner:
                lvm = inner["lockupViewModel"]
                raw_vid = lvm.get("contentId")
                if isinstance(raw_vid, str):
                    vid = raw_vid
                raw_title = (
                    lvm.get("metadata", {})
                    .get("lockupMetadataViewModel", {})
                    .get("title", {})
                    .get("content")
                )
                if isinstance(raw_title, str):
                    title = raw_title
            elif "videoRenderer" in inner:
                vr = inner["videoRenderer"]
                raw_vid = vr.get("videoId")
                if isinstance(raw_vid, str):
                    vid = raw_vid
                title_runs = vr.get("title", {}).get("runs", [])
                title = "".join(str(r.get("text", "")) for r in title_runs) or vr.get(
                    "title", {}
                ).get("simpleText")
        elif "videoRenderer" in item:
            vr = item["videoRenderer"]
            raw_vid = vr.get("videoId")
            if isinstance(raw_vid, str):
                vid = raw_vid
            title_runs = vr.get("title", {}).get("runs", [])
            title = "".join(str(r.get("text", "")) for r in title_runs) or vr.get(
                "title", {}
            ).get("simpleText")
        elif "gridVideoRenderer" in item:
            gvr = item["gridVideoRenderer"]
            raw_vid = gvr.get("videoId")
            if isinstance(raw_vid, str):
                vid = raw_vid
            title_runs = gvr.get("title", {}).get("runs", [])
            title = "".join(str(r.get("text", "")) for r in title_runs) or gvr.get(
                "title", {}
            ).get("simpleText")
        elif "playlistVideoRenderer" in item:
            pvr = item["playlistVideoRenderer"]
            raw_vid = pvr.get("videoId")
            if isinstance(raw_vid, str):
                vid = raw_vid
            title_runs = pvr.get("title", {}).get("runs", [])
            title = "".join(str(r.get("text", "")) for r in title_runs) or pvr.get(
                "title", {}
            ).get("simpleText")

        if vid and len(vid) == 11:
            res: dict[str, str] = {
                "url": f"https://www.youtube.com/watch?v={vid}",
                "id": vid,
            }
            if title:
                res["title"] = str(title)
            return res
        return None

    @staticmethod
    def _extract_continuation_token(item: dict[str, Any]) -> str | None:
        """Extract continuation token from an item if it contains continuationItemRenderer."""
        if "continuationItemRenderer" in item:
            cir = item["continuationItemRenderer"]
            endpoint = cir.get("continuationEndpoint", {})
            command = endpoint.get("continuationCommand", {})
            token = command.get("token")
            if token and isinstance(token, str):
                return str(token)
            btn_endpoint = (
                cir.get("button", {})
                .get("buttonRenderer", {})
                .get("command", {})
                .get("continuationCommand", {})
            )
            token = btn_endpoint.get("token")
            if token and isinstance(token, str):
                return str(token)
        return None

    @classmethod
    def _build_continuation_request(
        cls, channel: Any, token: str
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build the URL, headers, and payload for requesting the next page of videos."""
        res = channel._build_continuation_url(token)
        return str(res[0]), dict(res[1]), dict(res[2])

    @staticmethod
    def _extract_video_id(url: str) -> str | None:
        """Extract video ID from YouTube URL."""
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([^&\n?#]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None
