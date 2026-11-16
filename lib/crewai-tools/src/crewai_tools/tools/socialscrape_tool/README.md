# SocialScrape Tools

Tools for [SocialScrape](https://socialscrape.dev), a social media data API with one key and one JSON shape across 29 platforms: profiles, posts, comments, search, ad libraries and video transcripts.

SocialScrape maintains these tools (support@socialscrape.dev).

## Installation

```shell
uv add crewai-tools socialscrape-sdk
# or
pip install 'crewai-tools[socialscrape]'
```

Create an API key at https://socialscrape.dev/app and set it:

```shell
export SOCIALSCRAPE_API_KEY="..."
```

## Tools

| Tool | Inputs |
| --- | --- |
| `SocialScrapeProfileTool` | `platform`, one of `handle`, `url`, `id` |
| `SocialScrapePostsTool` | `platform`, one of `handle`, `url`, `id`, `cursor` |
| `SocialScrapePostTool` | `url`, or `platform` and `id` |
| `SocialScrapeCommentsTool` | `url`, or `platform` and `id`, `cursor` |
| `SocialScrapeTranscriptTool` | `url`, `language`, `allow_long_transcription`, `include_segments` |
| `SocialScrapeSearchTool` | `platform`, `query`, `cursor` |
| `SocialScrapeEndpointTool` | `endpoint` (any endpoint id, for example `facebook.ad_library.search`), `params` |

## Example

```python
from crewai import Agent
from crewai_tools import SocialScrapeProfileTool, SocialScrapeTranscriptTool

researcher = Agent(
    role="Social media researcher",
    goal="Summarise what a creator says and how large their audience is",
    backstory="You work from the data the tools return.",
    tools=[SocialScrapeProfileTool(), SocialScrapeTranscriptTool()],
)
```

## Options

- `api_key`: defaults to `SOCIALSCRAPE_API_KEY`.
- `max_age_days`: accept a cached result up to this many days old (0 to 30). Cached repeats are free.
- `trim`: return the compact response shape (default `True`).
- `max_output_chars`: cut long answers at this length (default 50000).

## Credits

Most requests cost 1 credit. Failed requests and cached repeats are free. Each answer includes `credits_charged` and `credits_remaining`. AI transcription of videos without captions costs more and needs `allow_long_transcription=True`.
