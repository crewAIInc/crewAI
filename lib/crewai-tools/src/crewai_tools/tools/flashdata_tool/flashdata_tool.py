"""Source-specific Google and YouTube tools for CrewAI agents."""

from typing import Any, ClassVar

from pydantic import BaseModel

from crewai_tools.tools.flashdata_tool._base import (
    FlashDataBaseTool,
    FlashDataDownloadSubmitTool,
    FlashDataRealtimeTool,
)
from crewai_tools.tools.flashdata_tool._inputs import (
    DownloadJobInput,
    GoogleInput,
    YouTubeCaptionsInput,
    YouTubeChannelInput,
    YouTubeDownloadInput,
    YouTubeExtendedSearchInput,
    YouTubeMetadataInput,
    YouTubeSearchInput,
    YouTubeSuggestionsInput,
    YouTubeTrainabilityInput,
    YouTubeTranscriptInput,
)


class FlashDataGoogleSearchTool(FlashDataRealtimeTool):
    """Find web evidence for an AI research agent. Search snippets are not fetched page content. Omitted count and locale are not filled in by the SDK. FlashData API calls use paid Credits."""

    name: str = "flashdata_GoogleSearch"
    description: str = "Find web evidence for an AI research agent. Search snippets are not fetched page content. Omitted count and locale are not filled in by the SDK. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = GoogleInput
    input_model: ClassVar[type[BaseModel]] = GoogleInput
    source: ClassVar[str] = "google_search"


class FlashDataGoogleNewsTool(FlashDataRealtimeTool):
    """Prepare a dated news briefing. Returns news search results, not article full text. FlashData API calls use paid Credits."""

    name: str = "flashdata_GoogleNews"
    description: str = "Prepare a dated news briefing. Returns news search results, not article full text. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = GoogleInput
    input_model: ClassVar[type[BaseModel]] = GoogleInput
    source: ClassVar[str] = "google_news"


class FlashDataGoogleImagesTool(FlashDataRealtimeTool):
    """Build an image research shortlist. Does not download images or establish image reuse rights. FlashData API calls use paid Credits."""

    name: str = "flashdata_GoogleImages"
    description: str = "Build an image research shortlist. Does not download images or establish image reuse rights. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = GoogleInput
    input_model: ClassVar[type[BaseModel]] = GoogleInput
    source: ClassVar[str] = "google_images"


class FlashDataGoogleShoppingTool(FlashDataRealtimeTool):
    """Prepare a product offer comparison. Prices are search snapshots; inventory and checkout price are not guaranteed. FlashData API calls use paid Credits."""

    name: str = "flashdata_GoogleShopping"
    description: str = "Prepare a product offer comparison. Prices are search snapshots; inventory and checkout price are not guaranteed. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = GoogleInput
    input_model: ClassVar[type[BaseModel]] = GoogleInput
    source: ClassVar[str] = "google_shopping"


class FlashDataGoogleVideosTool(FlashDataRealtimeTool):
    """Find video references across websites. Search discovers videos; it does not return video files or transcripts. FlashData API calls use paid Credits."""

    name: str = "flashdata_GoogleVideos"
    description: str = "Find video references across websites. Search discovers videos; it does not return video files or transcripts. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = GoogleInput
    input_model: ClassVar[type[BaseModel]] = GoogleInput
    source: ClassVar[str] = "google_videos"


class FlashDataGooglePlacesTool(FlashDataRealtimeTool):
    """Shortlist local businesses for a planning assistant. Contact details may be missing; verify current hours and availability with the business. FlashData API calls use paid Credits."""

    name: str = "flashdata_GooglePlaces"
    description: str = "Shortlist local businesses for a planning assistant. Contact details may be missing; verify current hours and availability with the business. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = GoogleInput
    input_model: ClassVar[type[BaseModel]] = GoogleInput
    source: ClassVar[str] = "google_places"


class FlashDataYouTubeSearchTool(FlashDataRealtimeTool):
    """Find YouTube tutorials for a learning agent. The requested count is a ceiling, not a guaranteed result count. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeSearch"
    description: str = "Find YouTube tutorials for a learning agent. The requested count is a ceiling, not a guaranteed result count. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeSearchInput
    input_model: ClassVar[type[BaseModel]] = YouTubeSearchInput
    source: ClassVar[str] = "youtube_search"


class FlashDataYouTubeSearchExtendedTool(FlashDataRealtimeTool):
    """Build a larger YouTube research candidate set. This is a distinct source with a separate higher rate than standard search. Requested count is not guaranteed. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeSearchExtended"
    description: str = "Build a larger YouTube research candidate set. This is a distinct source with a separate higher rate than standard search. Requested count is not guaranteed. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeExtendedSearchInput
    input_model: ClassVar[type[BaseModel]] = YouTubeExtendedSearchInput
    source: ClassVar[str] = "youtube_search_extended"


class FlashDataYouTubeMetadataTool(FlashDataRealtimeTool):
    """Inspect a video before adding it to a knowledge base. Metadata is not media download. Chapters and optional fields may be absent. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeMetadata"
    description: str = "Inspect a video before adding it to a knowledge base. Metadata is not media download. Chapters and optional fields may be absent. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeMetadataInput
    input_model: ClassVar[type[BaseModel]] = YouTubeMetadataInput
    source: ClassVar[str] = "youtube_metadata"


class FlashDataYouTubeTranscriptTool(FlashDataRealtimeTool):
    """Prepare timestamped video evidence for an agent. Reads an existing track. No ASR, translation, or automatic fallback to uploader captions. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeTranscript"
    description: str = "Prepare timestamped video evidence for an agent. Reads an existing track. No ASR, translation, or automatic fallback to uploader captions. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeTranscriptInput
    input_model: ClassVar[type[BaseModel]] = YouTubeTranscriptInput
    source: ClassVar[str] = "youtube_transcript"


class FlashDataYouTubeCaptionsTool(FlashDataRealtimeTool):
    """Review uploader captions for a content brief. Defaults to uploader captions. No ASR, translation, or automatic fallback to generated captions. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeCaptions"
    description: str = "Review uploader captions for a content brief. Defaults to uploader captions. No ASR, translation, or automatic fallback to generated captions. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeCaptionsInput
    input_model: ClassVar[type[BaseModel]] = YouTubeCaptionsInput
    source: ClassVar[str] = "youtube_captions"


class FlashDataYouTubeTrainabilityTool(FlashDataRealtimeTool):
    """Record a YouTube license-label signal. permitted is a license-label signal, not legal training permission or creator third-party training settings. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeTrainability"
    description: str = "Record a YouTube license-label signal. permitted is a license-label signal, not legal training permission or creator third-party training settings. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeTrainabilityInput
    input_model: ClassVar[type[BaseModel]] = YouTubeTrainabilityInput
    source: ClassVar[str] = "youtube_trainability"


class FlashDataYouTubeChannelTool(FlashDataRealtimeTool):
    """Prepare a channel and recent-video briefing. A recent-video snapshot is not the complete channel archive. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeChannel"
    description: str = "Prepare a channel and recent-video briefing. A recent-video snapshot is not the complete channel archive. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeChannelInput
    input_model: ClassVar[type[BaseModel]] = YouTubeChannelInput
    source: ClassVar[str] = "youtube_channel"


class FlashDataYouTubeSuggestionsTool(FlashDataRealtimeTool):
    """Expand topic ideas with YouTube autocomplete. Suggestions are not keyword volume or popularity measurements. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeSuggestions"
    description: str = "Expand topic ideas with YouTube autocomplete. Suggestions are not keyword volume or popularity measurements. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeSuggestionsInput
    input_model: ClassVar[type[BaseModel]] = YouTubeSuggestionsInput
    source: ClassVar[str] = "youtube_suggestions"


class FlashDataYouTubeDownloadSubmitTool(FlashDataDownloadSubmitTool):
    """Track a YouTube delivery to customer cloud storage. Paid asynchronous job. Requires customer storage. Save the receipt; never submit again to poll. Result JSON expires after 24 hours; media retention depends on storage. FlashData API calls use paid Credits."""

    name: str = "flashdata_YouTubeDownloadSubmit"
    description: str = "Track a YouTube delivery to customer cloud storage. Paid asynchronous job. Requires customer storage. Save the receipt; never submit again to poll. Result JSON expires after 24 hours; media retention depends on storage. FlashData API calls use paid Credits."
    args_schema: type[BaseModel] = YouTubeDownloadInput
    input_model: ClassVar[type[BaseModel]] = YouTubeDownloadInput


class FlashDataYouTubeDownloadStatusTool(FlashDataBaseTool):
    """Read a saved YouTube download job. Never submits, polls, or writes customer storage."""

    name: str = "flashdata_YouTubeDownloadStatus"
    description: str = "Read a saved YouTube download job. Never submits, polls, or writes customer storage."
    args_schema: type[BaseModel] = DownloadJobInput
    input_model: ClassVar[type[BaseModel]] = DownloadJobInput

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        data = self._validate_kwargs(kwargs)
        return self._client().status(data["job_id"])


class FlashDataYouTubeDownloadResultTool(FlashDataBaseTool):
    """Read a saved YouTube download job. Never submits, polls, or writes customer storage."""

    name: str = "flashdata_YouTubeDownloadResult"
    description: str = "Read a saved YouTube download job. Never submits, polls, or writes customer storage."
    args_schema: type[BaseModel] = DownloadJobInput
    input_model: ClassVar[type[BaseModel]] = DownloadJobInput

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        data = self._validate_kwargs(kwargs)
        return self._client().result(data["job_id"])
