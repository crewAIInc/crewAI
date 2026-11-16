from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.socialscrape_tool.socialscrape_base_tool import (
    SocialScrapeBaseTool,
    platform_from_url,
)


# Platform to the endpoint id that does the job. Each id is a path on
# https://api.socialscrape.dev: "tiktok.profile" is GET /v1/tiktok/profile.
PROFILE_ENDPOINTS: dict[str, str] = {
    "apple_music": "apple_music.artist",
    "bluesky": "bluesky.profile",
    "cosmos": "cosmos.profile",
    "facebook": "facebook.profile",
    "github": "github.user",
    "instagram": "instagram.profile",
    "kwai": "kwai.profile",
    "linkedin": "linkedin.profile",
    "snapchat": "snapchat.profile",
    "soundcloud": "soundcloud.artist",
    "spotify": "spotify.artist",
    "threads": "threads.profile",
    "tiktok": "tiktok.profile",
    "truthsocial": "truthsocial.profile",
    "twitch": "twitch.profile",
    "twitter": "twitter.profile",
    "youtube": "youtube.channel",
}
POSTS_ENDPOINTS: dict[str, str] = {
    "bluesky": "bluesky.profile.posts",
    "cosmos": "cosmos.profile.posts",
    "facebook": "facebook.profile.posts",
    "instagram": "instagram.profile.posts",
    "kwai": "kwai.profile.posts",
    "linkedin": "linkedin.company.posts",
    "rumble": "rumble.channel.videos",
    "telegram": "telegram.channel.posts",
    "threads": "threads.profile.posts",
    "tiktok": "tiktok.profile.videos",
    "truthsocial": "truthsocial.profile.posts",
    "twitch": "twitch.user.videos",
    "twitter": "twitter.profile.tweets",
    "youtube": "youtube.channel.videos",
}
POST_ENDPOINTS: dict[str, str] = {
    "bluesky": "bluesky.post",
    "cosmos": "cosmos.post",
    "facebook": "facebook.post",
    "instagram": "instagram.post",
    "kick": "kick.clip",
    "kwai": "kwai.post",
    "linkedin": "linkedin.post",
    "pinterest": "pinterest.pin",
    "reddit": "reddit.post",
    "rumble": "rumble.video",
    "snapchat": "snapchat.spotlight",
    "telegram": "telegram.post",
    "threads": "threads.post",
    "tiktok": "tiktok.video",
    "truthsocial": "truthsocial.post",
    "twitch": "twitch.clip",
    "twitter": "twitter.tweet",
    "youtube": "youtube.video",
}
COMMENTS_ENDPOINTS: dict[str, str] = {
    "facebook": "facebook.post.comments",
    "instagram": "instagram.post.comments",
    "reddit": "reddit.post.comments",
    "rumble": "rumble.video.comments",
    "snapchat": "snapchat.spotlight.comments",
    "tiktok": "tiktok.video.comments",
    "youtube": "youtube.video.comments",
}
SEARCH_ENDPOINTS: dict[str, str] = {
    "apple_music": "apple_music.search",
    "cosmos": "cosmos.search",
    "facebook": "facebook.search.videos",
    "google": "google.search",
    "instagram": "instagram.search.reels",
    "linkedin": "linkedin.search.posts",
    "pinterest": "pinterest.search",
    "reddit": "reddit.search",
    "rumble": "rumble.search",
    "spotify": "spotify.search",
    "threads": "threads.search",
    "tiktok": "tiktok.search.keyword",
    "twitter": "twitter.search.posts",
    "youtube": "youtube.search",
}
# Link hosts, used to read the platform from a post URL.
POST_HOSTS: dict[str, tuple[str, ...]] = {
    "bluesky": ("bsky.app",),
    "cosmos": ("cosmos.so",),
    "facebook": ("facebook.com", "fb.com", "fb.watch"),
    "instagram": ("instagram.com", "instagr.am"),
    "kick": ("kick.com",),
    "kwai": ("kwai.com",),
    "linkedin": ("linkedin.com",),
    "pinterest": ("pinterest.com", "pin.it"),
    "reddit": ("reddit.com", "redd.it"),
    "rumble": ("rumble.com",),
    "snapchat": ("snapchat.com",),
    "telegram": ("t.me", "telegram.me"),
    "threads": ("threads.net", "threads.com"),
    "tiktok": ("tiktok.com",),
    "truthsocial": ("truthsocial.com",),
    "twitch": ("twitch.tv",),
    "twitter": ("x.com", "twitter.com"),
    "youtube": ("youtube.com", "youtu.be"),
}
TRANSCRIPT_ENDPOINT = "transcript"

_CURSOR = (
    "The `cursor` of the previous answer, to get the next page. "
    "Leave it out for the first page."
)


def _platform(value: str, table: dict[str, str], job: str) -> str:
    key = value.strip().lower().replace("-", "_")
    key = {"x": "twitter"}.get(key, key)
    if key not in table:
        raise ValueError(
            f"No {job} lookup for {value!r}. Platforms: {', '.join(sorted(table))}"
        )
    return key


def _choices(table: dict[str, str]) -> str:
    return ", ".join(sorted(table))


class SocialScrapeProfileToolSchema(BaseModel):
    platform: str = Field(
        description=f"Platform key. One of: {_choices(PROFILE_ENDPOINTS)}."
    )
    handle: str | None = Field(
        default=None, description="Username or handle, with or without the leading @."
    )
    url: str | None = Field(
        default=None, description="Full URL of the profile, channel or page."
    )
    id: str | None = Field(
        default=None, description="The platform's own id of the account, if it has one."
    )


class SocialScrapePostsToolSchema(SocialScrapeProfileToolSchema):
    platform: str = Field(
        description=f"Platform key. One of: {_choices(POSTS_ENDPOINTS)}."
    )
    cursor: str | None = Field(default=None, description=_CURSOR)


class SocialScrapePostToolSchema(BaseModel):
    url: str | None = Field(
        default=None,
        description="Full URL of the post, video, tweet, pin or clip. The platform is read from it.",
    )
    platform: str | None = Field(
        default=None,
        description=f"Platform key, needed only with `id`. One of: {_choices(POST_ENDPOINTS)}.",
    )
    id: str | None = Field(default=None, description="The post's id. Needs `platform`.")


class SocialScrapeCommentsToolSchema(BaseModel):
    url: str | None = Field(
        default=None,
        description="Full URL of the post or video. The platform is read from it.",
    )
    platform: str | None = Field(
        default=None,
        description=f"Platform key, needed only with `id`. One of: {_choices(COMMENTS_ENDPOINTS)}.",
    )
    id: str | None = Field(default=None, description="The post's id. Needs `platform`.")
    cursor: str | None = Field(default=None, description=_CURSOR)


class SocialScrapeTranscriptToolSchema(BaseModel):
    url: str = Field(description="Full URL of the video.")
    language: str | None = Field(
        default=None,
        description="Preferred caption language as a BCP 47 code, for example en or pt-BR.",
    )
    allow_long_transcription: bool = Field(
        default=False,
        description=(
            "Allow AI transcription of audio longer than 2 minutes when the video has "
            "no captions, at 5 credits per started minute. Without it such a video "
            "answers an error that names the cost, which is free."
        ),
    )
    include_segments: bool = Field(
        default=False,
        description="Also return the timed segments (start, end, text).",
    )


class SocialScrapeSearchToolSchema(BaseModel):
    platform: str = Field(
        description=f"Platform key. One of: {_choices(SEARCH_ENDPOINTS)}."
    )
    query: str = Field(description="Search terms.")
    cursor: str | None = Field(default=None, description=_CURSOR)


class SocialScrapeEndpointToolSchema(BaseModel):
    endpoint: str = Field(
        description=(
            "Endpoint id, for example youtube.channel.videos or facebook.ad_library.search. "
            "The list is in the API reference at https://socialscrape.dev/docs/api."
        )
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description='The endpoint\'s parameters by name, for example {"handle": "nasa"}.',
    )


def _identifier(
    handle: str | None, url: str | None, id: str | None
) -> dict[str, str | None]:
    given = {"handle": handle, "url": url, "id": id}
    if not any(given.values()):
        raise ValueError("Pass one of handle, url or id.")
    return given


class SocialScrapeProfileTool(SocialScrapeBaseTool):
    """Get a profile, channel or artist from a social platform with SocialScrape."""

    name: str = "SocialScrape profile"
    description: str = (
        "Get a profile from a social platform: name, bio, links, verification and "
        f"follower counts. Platforms: {_choices(PROFILE_ENDPOINTS)}. Pass the platform "
        "and a handle, url or id. Costs 1 credit; a failed lookup is free."
    )
    args_schema: type[BaseModel] = SocialScrapeProfileToolSchema

    def _run(
        self,
        platform: str,
        handle: str | None = None,
        url: str | None = None,
        id: str | None = None,
        **_: Any,
    ) -> str:
        try:
            endpoint = PROFILE_ENDPOINTS[
                _platform(platform, PROFILE_ENDPOINTS, "profile")
            ]
            params = _identifier(handle, url, id)
        except ValueError as e:
            return f"Error: {e}"
        return self._request(endpoint, params)


class SocialScrapePostsTool(SocialScrapeBaseTool):
    """List the latest posts or videos of a profile, channel or page with SocialScrape."""

    name: str = "SocialScrape posts"
    description: str = (
        "List the latest posts or videos of a profile, channel or page, newest first, "
        "one page per call: text, time, stats and media links. "
        f"Platforms: {_choices(POSTS_ENDPOINTS)}. Pass the platform and a handle, url or id. "
        "When the answer has `has_more` true, pass its `cursor` back for the next page. "
        "Costs 1 credit per page; a failed call is free."
    )
    args_schema: type[BaseModel] = SocialScrapePostsToolSchema

    def _run(
        self,
        platform: str,
        handle: str | None = None,
        url: str | None = None,
        id: str | None = None,
        cursor: str | None = None,
        **_: Any,
    ) -> str:
        try:
            endpoint = POSTS_ENDPOINTS[_platform(platform, POSTS_ENDPOINTS, "posts")]
            params = {**_identifier(handle, url, id), "cursor": cursor}
        except ValueError as e:
            return f"Error: {e}"
        return self._request(endpoint, params)


class SocialScrapePostTool(SocialScrapeBaseTool):
    """Get one post, video, tweet, pin or clip by its URL with SocialScrape."""

    name: str = "SocialScrape post"
    description: str = (
        "Get one post, video, tweet, pin or clip: text, author, time, stats and media "
        f"links. Platforms: {_choices(POST_ENDPOINTS)}. Pass its url; the platform is "
        "read from the link. Costs 1 credit; a failed lookup is free."
    )
    args_schema: type[BaseModel] = SocialScrapePostToolSchema

    def _run(
        self,
        url: str | None = None,
        platform: str | None = None,
        id: str | None = None,
        **_: Any,
    ) -> str:
        try:
            endpoint, params = _post_request(
                POST_ENDPOINTS, "post", url=url, platform=platform, id=id
            )
        except ValueError as e:
            return f"Error: {e}"
        return self._request(endpoint, params)


class SocialScrapeCommentsTool(SocialScrapeBaseTool):
    """List the comments on a post or video with SocialScrape."""

    name: str = "SocialScrape comments"
    description: str = (
        "List the comments on a post or video, one page per call: text, author, time "
        f"and likes per comment. Platforms: {_choices(COMMENTS_ENDPOINTS)}. Pass the "
        "post's url; the platform is read from the link. When the answer has `has_more` "
        "true, pass its `cursor` back for the next page. Costs 1 credit per page; a "
        "failed call is free."
    )
    args_schema: type[BaseModel] = SocialScrapeCommentsToolSchema

    def _run(
        self,
        url: str | None = None,
        platform: str | None = None,
        id: str | None = None,
        cursor: str | None = None,
        **_: Any,
    ) -> str:
        try:
            endpoint, params = _post_request(
                COMMENTS_ENDPOINTS, "comments", url=url, platform=platform, id=id
            )
        except ValueError as e:
            return f"Error: {e}"
        return self._request(endpoint, {**params, "cursor": cursor})


def _post_request(
    table: dict[str, str],
    job: str,
    *,
    url: str | None,
    platform: str | None,
    id: str | None,
) -> tuple[str, dict[str, str | None]]:
    """Pick the endpoint for a post or its comments and the one identifier to send."""
    if not platform:
        if not url:
            raise ValueError(
                f"Pass a url, or a platform and an id. Platforms: {_choices(table)}"
            )
        found = platform_from_url(
            url, {k: v for k, v in POST_HOSTS.items() if k in table}
        )
        if found is None:
            raise ValueError(
                f"No platform recognised in {url!r}. Pass platform, one of: {_choices(table)}"
            )
        platform = found
    endpoint = table[_platform(platform, table, job)]
    if url:
        return endpoint, {"url": url}
    if id:
        return endpoint, {"id": id}
    raise ValueError("Pass a url, or an id with its platform.")


class SocialScrapeTranscriptTool(SocialScrapeBaseTool):
    """Get the transcript of a video from its URL with SocialScrape."""

    name: str = "SocialScrape transcript"
    description: str = (
        "Get the transcript of a video from its URL: the full text, its language and "
        "where it came from (captions, or AI transcription of the audio when there are "
        "none). Works with YouTube, TikTok, Instagram and other video links. Costs 1 "
        "credit from captions, and 1 credit for AI transcription up to 2 minutes of "
        "audio. Longer audio needs allow_long_transcription and costs 5 credits per "
        "started minute. A failed call is free."
    )
    args_schema: type[BaseModel] = SocialScrapeTranscriptToolSchema

    def _run(
        self,
        url: str,
        language: str | None = None,
        allow_long_transcription: bool = False,
        include_segments: bool = False,
        **_: Any,
    ) -> str:
        return self._request(
            TRANSCRIPT_ENDPOINT,
            {
                "url": url,
                "language": language,
                "allow_long_transcription": True if allow_long_transcription else None,
            },
            include_segments=include_segments,
        )

    def _shape(self, data: Any, include_segments: bool = False, **_: Any) -> Any:
        if isinstance(data, dict) and not include_segments:
            return {k: v for k, v in data.items() if k != "segments"}
        return data


class SocialScrapeSearchTool(SocialScrapeBaseTool):
    """Search one platform by keyword with SocialScrape."""

    name: str = "SocialScrape search"
    description: str = (
        "Search one platform by keyword, one page per call. Posts and videos on most "
        "platforms, tracks and artists on Spotify and Apple Music, web results on "
        f"Google. Platforms: {_choices(SEARCH_ENDPOINTS)}. When the answer has "
        "`has_more` true, pass its `cursor` back for the next page. Costs 1 credit per "
        "page; a failed call is free."
    )
    args_schema: type[BaseModel] = SocialScrapeSearchToolSchema

    def _run(
        self, platform: str, query: str, cursor: str | None = None, **_: Any
    ) -> str:
        try:
            endpoint = SEARCH_ENDPOINTS[_platform(platform, SEARCH_ENDPOINTS, "search")]
        except ValueError as e:
            return f"Error: {e}"
        return self._request(endpoint, {"query": query, "cursor": cursor})


class SocialScrapeEndpointTool(SocialScrapeBaseTool):
    """Run any SocialScrape data endpoint by its id."""

    name: str = "SocialScrape endpoint"
    description: str = (
        "Run any SocialScrape endpoint by its id, for example youtube.channel.videos "
        "or facebook.ad_library.search, with its parameters by name. Covers ad "
        "libraries, followers, hashtags, playlists, music, shops and more than 190 "
        "other endpoints. Most cost 1 credit per call; a failed call is free. If a "
        "parameter is wrong the answer names what the endpoint takes."
    )
    args_schema: type[BaseModel] = SocialScrapeEndpointToolSchema

    def _run(
        self, endpoint: str, params: dict[str, Any] | None = None, **_: Any
    ) -> str:
        try:
            return self._request(endpoint, dict(params or {}))
        except ValueError as e:
            return f"Error: {e}"


__all__ = [
    "SocialScrapeCommentsTool",
    "SocialScrapeEndpointTool",
    "SocialScrapePostTool",
    "SocialScrapePostsTool",
    "SocialScrapeProfileTool",
    "SocialScrapeSearchTool",
    "SocialScrapeTranscriptTool",
]
