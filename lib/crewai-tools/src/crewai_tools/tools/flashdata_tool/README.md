# FlashData tools

Seventeen native CrewAI tools cover fifteen Google and YouTube data sources. Each source has its own input schema, task guide and website setup link. YouTube delivery uses separate submit/status/result tools.

## Install

These additions are not in the published `crewai-tools` release. Use Python 3.10–3.13 and the pinned source revision (CrewAI 1.15.28):

```shell
uv add "crewai-tools @ git+https://github.com/rocosen/crewAI.git@449d1f0f70ed312fb3fc371b7df9f97386ef326f#subdirectory=lib/crewai-tools"
```

Create an account and API key using the [Google Search source guide](https://flashdata.dev/docs/sources/google_search), then set `FLASHDATA_API_KEY` through your secret manager. Installation is free under MIT; API requests consume paid FlashData Credits. Model and customer cloud-storage charges are separate.

```python
from crewai_tools import FlashDataGoogleSearchTool

tool = FlashDataGoogleSearchTool(max_usage_count=1)
print(tool.args_schema.model_json_schema())  # No API or model call.
# Attach tool to your existing Agent, or call tool.run(query="your research question").
```

The guides include direct calls, actual Agent/Task/Crew examples, source inputs, preserved JSON outputs, costs and limitations. Agent examples require a configured model and can incur model and API charges. No model, embedding service or vector store is included.

| Source | Tools | Guide |
| --- | --- | --- |
| `google_search` | `FlashDataGoogleSearchTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/google-search-flashdata.mdx) |
| `google_news` | `FlashDataGoogleNewsTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/google-news-flashdata.mdx) |
| `google_images` | `FlashDataGoogleImagesTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/google-images-flashdata.mdx) |
| `google_shopping` | `FlashDataGoogleShoppingTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/google-shopping-flashdata.mdx) |
| `google_videos` | `FlashDataGoogleVideosTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/google-videos-flashdata.mdx) |
| `google_places` | `FlashDataGooglePlacesTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/google-places-flashdata.mdx) |
| `youtube_search` | `FlashDataYouTubeSearchTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-search-flashdata.mdx) |
| `youtube_search_extended` | `FlashDataYouTubeSearchExtendedTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-search-extended-flashdata.mdx) |
| `youtube_metadata` | `FlashDataYouTubeMetadataTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-metadata-flashdata.mdx) |
| `youtube_transcript` | `FlashDataYouTubeTranscriptTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-transcript-flashdata.mdx) |
| `youtube_captions` | `FlashDataYouTubeCaptionsTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-captions-flashdata.mdx) |
| `youtube_trainability` | `FlashDataYouTubeTrainabilityTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-trainability-flashdata.mdx) |
| `youtube_channel` | `FlashDataYouTubeChannelTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-channel-flashdata.mdx) |
| `youtube_suggestions` | `FlashDataYouTubeSuggestionsTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-suggestions-flashdata.mdx) |
| `youtube_download` | `FlashDataYouTubeDownloadSubmitTool`, `FlashDataYouTubeDownloadStatusTool`, `FlashDataYouTubeDownloadResultTool` | [Task guide](../../../../../../docs/edge/en/tools/search-research/youtube-download-flashdata.mdx) |

## Runtime behavior

- Credentials are read from environment variables at call time. `api_key_env` and `storage_env` configure variable names, not values. Tool argument schemas do not expose secrets, source overrides or submission authorization.
- `run`/`arun` and CrewAI structured invocation are supported. Serialize with `tool.model_dump()` and restore using the same concrete tool class’s `model_validate`; secret values are never serialized. Live results are not cached.
- The HTTP client has no automatic retry, pagination or polling. Network timeout is 120 seconds (15 to connect), with a 20 MiB response limit. A submission timeout may have an unknown billing outcome; inspect Jobs before retrying.
- Direct calls return API JSON. Download lifecycle responses filter fields that could contain credentials. Completed result manifests can contain signed URLs and should be shared only with intended recipients.
- Download submission is disabled by default. `allow_submit=True` and a valid `FLASHDATA_DOWNLOAD_STORAGE` JSON credential are operator configuration. Status/result reads never submit. Result fetches validate the management API’s trusted HTTPS origin and do not forward the API key.
- `max_usage_count` is CrewAI’s tool-instance use limit; it is not an account budget. Keep cloud-write tools out of an Agent that only needs read access.

## Tests and provenance

The source-specific suite uses real CrewAI tool/structured-tool objects with mocked HTTP and disabled network. It does not exercise a paid model, live FlashData API or customer storage.

The API input and storage contracts adapt FlashData’s MIT-licensed [Haystack integration](https://github.com/rocosen/flashdata-mcp/tree/haystack-v0.1.0/packages/haystack). The CrewAI adapter, examples and tests are native to this repository. Preserve the upstream MIT notice in [FLASHDATA-LICENSE](FLASHDATA-LICENSE).
