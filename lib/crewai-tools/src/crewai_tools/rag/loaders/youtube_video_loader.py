"""YouTube video loader for extracting transcripts from YouTube videos."""

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class YoutubeVideoLoader(BaseLoader):
    """Loader for YouTube videos."""

    def load(self, source: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        """Load and extract transcript from a YouTube video.

        Args:
            source: The source content containing the YouTube URL

        Returns:
            LoaderResult with transcript content

        Raises:
            ImportError: If required YouTube libraries aren't installed
            ValueError: If the URL is not a valid YouTube video URL
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as e:
            raise ImportError(
                "YouTube support requires youtube-transcript-api. "
                "Install with: uv add youtube-transcript-api"
            ) from e

        video_url = source.source
        video_id = self._extract_video_id(video_url)

        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")

        metadata: dict[str, Any] = {
            "source": video_url,
            "video_id": video_id,
            "data_type": "youtube_video",
        }

        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)

            try:
                transcript = transcript_list.find_transcript(["en"])
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(["en"])
                except Exception:
                    transcript = next(iter(transcript_list))

            if transcript:
                metadata["language"] = transcript.language
                metadata["is_generated"] = transcript.is_generated

                transcript_data = transcript.fetch()

                text_content = []
                for entry in transcript_data:
                    text = entry.text.strip() if hasattr(entry, "text") else ""
                    if text:
                        text_content.append(text)

                content = " ".join(text_content)

                try:
                    from pytube import YouTube  # type: ignore[import-untyped]

                    yt = YouTube(video_url)
                    metadata["title"] = yt.title
                    metadata["author"] = yt.author
                    metadata["length_seconds"] = yt.length
                    metadata["description"] = (
                        yt.description[:500] if yt.description else None
                    )

                    if yt.title:
                        content = f"Title: {yt.title}\n\nAuthor: {yt.author or 'Unknown'}\n\nTranscript:\n{content}"
                except Exception:  # noqa: S110
                    pass
            else:
                raise ValueError(
                    f"No transcript available for YouTube video: {video_id}"
                )

        except Exception as e:
            raise ValueError(
                f"Unable to extract transcript from YouTube video {video_id}: {e!s}"
            ) from e

        return LoaderResult(
            content=content,
            source=video_url,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=video_url, content=content),
        )

    @staticmethod
    def _extract_video_id(url: str) -> str | None:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([^&\n?#]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if hostname:
                hostname_lower = hostname.lower()
                # Allow youtube.com and any subdomain of youtube.com, plus youtu.be shortener
                if (
                    hostname_lower == "youtube.com"
                    or hostname_lower.endswith(".youtube.com")
                    or hostname_lower == "youtu.be"
                ):
                    query_params = parse_qs(parsed.query)
                    if "v" in query_params:
                        return query_params["v"][0]
        except Exception:  # noqa: S110
            pass

        return None
