"""Shared context-window lookup for CrewAI LLMs.

Limits come from a checked-in snapshot of LiteLLM's model catalog:
https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

``resolve_context_window_size`` applies the usable context ratio, so every
call site shares one matching rule.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
import concurrent.futures
import contextvars
from importlib.resources import files
import json
import re
from typing import TYPE_CHECKING, Any, Final, cast

from crewai_core.printer import PRINTER

from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.i18n import I18N_DEFAULT
from crewai.utilities.message_utils import (
    format_message_for_llm,
    message_content_text,
    messages_to_text,
)


if TYPE_CHECKING:
    from crewai.utilities.types import LLMMessage

DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192
CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85
COMPACTION_RETRY_BUDGET_RATIO: Final[float] = 0.75
_COMPACTION_MAX_RETRIES: Final[int] = 3

_CATALOG_FILENAME: Final[str] = "model_prices_and_context_window.json"
_MAX_SUMMARY_LAYERS: Final[int] = 1
_OPENAI_PROVIDERS: Final[frozenset[str]] = frozenset({"openai"})
_AZURE_PROVIDERS: Final[frozenset[str]] = frozenset({"azure", "azure_ai"})
_ANTHROPIC_PROVIDERS: Final[frozenset[str]] = frozenset({"anthropic"})
_GEMINI_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"gemini", "vertex_ai-language-models"}
)
_BEDROCK_PROVIDERS: Final[frozenset[str]] = frozenset({"bedrock", "bedrock_converse"})
_REGION_PREFIXES: Final[tuple[str, ...]] = (
    "us-gov.",
    "global.",
    "apac.",
    "us.",
    "eu.",
    "au.",
    "jp.",
)
_VENDOR_PREFIXES: Final[tuple[str, ...]] = (
    "anthropic.",
    "amazon.",
    "meta.",
    "google.",
    "openai.",
    "mistral.",
    "cohere.",
    "deepseek.",
    "qwen.",
    "ai21.",
    "nvidia.",
    "moonshotai.",
    "moonshot.",
    "xai.",
    "zai.",
    "writer.",
    "twelvelabs.",
    "minimax.",
)
_DATE_IN_MODEL_ID: Final[re.Pattern[str]] = re.compile(
    r"-\d{4}-\d{2}-\d{2}|-\d{8}|@\d{8}"
)
_TRAILING_VERSION: Final[re.Pattern[str]] = re.compile(r"(?P<body>.+)-v\d+$")


def context_windows_from_catalog(
    catalog: Mapping[str, Any],
    *,
    providers: frozenset[str] | None = None,
) -> dict[str, int]:
    """Build model-id context windows from a LiteLLM catalog payload.

    Catalog keys are kept as published. When a key is ``provider/model`` and
    that provider matches ``litellm_provider``, the bare model id is also
    indexed so native providers can look up the id they send to the API.
    That alias is omitted when those entries disagree on the limit, so lookup
    can fall through to the bare-name index, which keeps the most common limit.
    Exact catalog keys override aliases that do agree.
    """
    alias_limits: dict[str, set[int]] = {}
    exact: dict[str, int] = {}
    for name, props in catalog.items():
        if name == "sample_spec" or not isinstance(props, dict):
            continue
        provider = props.get("litellm_provider")
        if providers is not None and provider not in providers:
            continue
        limit = props.get("max_input_tokens")
        if isinstance(limit, bool) or not isinstance(limit, int):
            continue
        exact[name] = limit
        provider_name, separator, remainder = name.partition("/")
        if separator and remainder and provider_name == provider:
            alias_limits.setdefault(remainder, set()).add(limit)
    aliases = {
        name: next(iter(limits))
        for name, limits in alias_limits.items()
        if len(limits) == 1
    }
    return {**aliases, **exact}


def _load_catalog() -> dict[str, Any]:
    resource = files("crewai.llms").joinpath("litellm_catalog", _CATALOG_FILENAME)
    catalog = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("LiteLLM context catalog must be a JSON object")
    return catalog


def _catalog_provider_prefixes(catalog: Mapping[str, Any]) -> frozenset[str]:
    """Return provider prefixes that catalog keys actually use."""
    prefixes: set[str] = set()
    for name, props in catalog.items():
        if not isinstance(props, dict):
            continue
        provider = props.get("litellm_provider")
        if not isinstance(provider, str):
            continue
        prefix, separator, remainder = name.partition("/")
        if separator and remainder and prefix == provider:
            prefixes.add(prefix.lower())
    return frozenset(prefixes)


_CATALOG = _load_catalog()
_CATALOG_PROVIDER_PREFIXES: Final[frozenset[str]] = _catalog_provider_prefixes(_CATALOG)
LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = context_windows_from_catalog(_CATALOG)
OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_OPENAI_PROVIDERS
)
AZURE_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_AZURE_PROVIDERS
)
AZURE_OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    **OPENAI_CONTEXT_WINDOWS,
    **AZURE_CONTEXT_WINDOWS,
}
ANTHROPIC_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_ANTHROPIC_PROVIDERS
)
GEMINI_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_GEMINI_PROVIDERS
)
BEDROCK_CONTEXT_WINDOWS: Final[dict[str, int]] = context_windows_from_catalog(
    _CATALOG, providers=_BEDROCK_PROVIDERS
)


def bare_model_name(name: str) -> str:
    """Return the model family without provider, region, date, or version suffix.

    ``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` and
    ``us.anthropic.claude-sonnet-4-5-v1:0`` both become ``claude-sonnet-4-5``.
    A trailing ``-vN`` is removed only when a model family remains, so
    ``claude-v2`` stays intact.
    """
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    for prefix in _REGION_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for prefix in _VENDOR_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    name = _DATE_IN_MODEL_ID.sub("", name)
    while True:
        stem, separator, segment = name.rpartition(":")
        if not separator or not segment:
            break
        name = stem
    version = _TRAILING_VERSION.fullmatch(name)
    if version is not None and "-" in version.group("body"):
        name = version.group("body")
    return name


def _bare_lookup_name(name: str) -> str:
    """Return the bare-index key for a model id.

    A slash prefix is removed only when that prefix is a catalog provider.
    ``acme/gpt-5.6-luna`` stays intact so it does not inherit ``gpt-5.6-luna``.
    """
    prefix, separator, remainder = name.partition("/")
    if separator and remainder and prefix.lower() not in _CATALOG_PROVIDER_PREFIXES:
        return name
    return bare_model_name(name)


def bare_context_windows(sizes: Mapping[str, int]) -> dict[str, int]:
    """Index context limits by bare model name.

    Each catalog entry casts a vote for its limit. The bare name keeps the
    limit with the most votes. A tie keeps the smaller limit.
    """
    votes_by_name: dict[str, Counter[int]] = {}
    for name, limit in sizes.items():
        bare_name = bare_model_name(name)
        if not bare_name:
            continue
        votes_by_name.setdefault(bare_name, Counter()).update((limit,))
    windows: dict[str, int] = {}
    for name, votes in votes_by_name.items():
        ranking = votes.most_common()
        top_votes = ranking[0][1]
        windows[name] = min(limit for limit, count in ranking if count == top_votes)
    return windows


BARE_CONTEXT_WINDOWS: Final[dict[str, int]] = bare_context_windows(
    LLM_CONTEXT_WINDOW_SIZES
)


def resolve_context_window_size(
    model: str,
    sizes: Mapping[str, int],
    *,
    default: int,
    extra_names: Sequence[str] = (),
) -> int:
    """Return the usable context window for a catalog model id.

    Exact IDs are resolved in ``sizes``. A miss falls back to
    ``BARE_CONTEXT_WINDOWS``, which is built from the full catalog rather than
    the provider map. An unknown ``provider/model`` prefix is not reduced to
    the bare model.
    """
    for name in (model, *extra_names):
        size = sizes.get(name)
        if size is not None:
            return int(size * CONTEXT_WINDOW_USAGE_RATIO)

    for name in (model, *extra_names):
        size = BARE_CONTEXT_WINDOWS.get(_bare_lookup_name(name))
        if size is not None:
            return int(size * CONTEXT_WINDOW_USAGE_RATIO)
    return int(default * CONTEXT_WINDOW_USAGE_RATIO)


def summarize_messages(
    llm: Any,
    messages: list[LLMMessage],
    callbacks: list[Any],
    verbose: bool,
) -> None:
    """Compact ``messages`` in place using ``llm``'s context-window contract."""
    system_messages = [
        message for message in messages if message.get("role") == "system"
    ]
    work_messages = [message for message in messages if message.get("role") != "system"]
    if not work_messages:
        return

    preserved_files = _collect_summary_files(messages)
    max_tokens = llm.get_context_window_size()
    chunks = _chunk_messages(llm, work_messages, max_tokens)
    summaries = _summarize_chunks(llm, chunks, callbacks, verbose)
    summaries = _compact_summaries(
        llm, system_messages, summaries, callbacks, verbose, max_tokens
    )
    summary_message = format_message_for_llm(
        I18N_DEFAULT.slice("summary").format(merged_summary="\n\n".join(summaries))
    )
    if preserved_files:
        summary_message["files"] = preserved_files

    messages.clear()
    messages.extend(system_messages)
    messages.append(summary_message)


def _collect_summary_files(messages: list[LLMMessage]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for message in messages:
        if message.get("role") == "user" and message.get("files"):
            files.update(message["files"])
    return files


def _summarize_chunks(
    llm: Any,
    chunks: list[list[LLMMessage]],
    callbacks: list[Any],
    verbose: bool,
) -> list[str]:
    """Summarize chunks concurrently, preserving their order."""
    if verbose:
        if len(chunks) <= 1:
            PRINTER.print(content=f"Summarizing 1/{len(chunks)}...", color="yellow")
        else:
            PRINTER.print(
                content=f"Summarizing {len(chunks)} chunks in parallel...",
                color="yellow",
            )

    async def summarize_chunk(
        chunk: list[LLMMessage], max_tokens: int, retries: int = 0
    ) -> str:
        try:
            async with semaphore:
                response = str(
                    await llm.acall(_build_summary_prompt(chunk), callbacks=callbacks)
                )
        except Exception as error:
            if not LLMContextLengthExceededError(
                str(error)
            )._is_context_length_exceeded_error(error):
                raise
            if retries >= _COMPACTION_MAX_RETRIES:
                raise LLMContextLengthExceededError(
                    "Summarization failed after exhausting context-length retries."
                ) from None

            retry_budget = max(1, int(max_tokens * COMPACTION_RETRY_BUDGET_RATIO))
            while retry_budget < max_tokens:
                sub_chunks = _chunk_messages(llm, chunk, retry_budget)
                if len(sub_chunks) > 1:
                    return "\n\n".join(
                        await asyncio.gather(
                            *(
                                summarize_chunk(item, retry_budget, retries + 1)
                                for item in sub_chunks
                            )
                        )
                    )
                max_tokens, retry_budget = (
                    retry_budget,
                    max(1, int(retry_budget * COMPACTION_RETRY_BUDGET_RATIO)),
                )

            raise LLMContextLengthExceededError(
                "Summarization chunk could not be split further."
            ) from None

        match = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    semaphore = asyncio.Semaphore(4)

    async def gather_summaries() -> list[str]:
        max_tokens = llm.get_context_window_size()
        return list(
            await asyncio.gather(
                *(summarize_chunk(chunk, max_tokens) for chunk in chunks)
            )
        )

    coroutine = gather_summaries()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    context = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(context.run, asyncio.run, coroutine).result()


def _compact_summaries(
    llm: Any,
    system_messages: list[LLMMessage],
    summaries: list[str],
    callbacks: list[Any],
    verbose: bool,
    max_tokens: int,
) -> list[str]:
    """Condense summaries once more when the replacement history does not fit."""
    for layer in range(_MAX_SUMMARY_LAYERS + 1):
        replacement = format_message_for_llm(
            I18N_DEFAULT.slice("summary").format(merged_summary="\n\n".join(summaries))
        )
        if (
            int(llm.get_input_tokens_count([*system_messages, replacement]))
            <= max_tokens
        ):
            return summaries
        if layer == _MAX_SUMMARY_LAYERS:
            break

        summary_messages = [format_message_for_llm(summary) for summary in summaries]
        chunks = _chunk_messages(llm, summary_messages, max_tokens)
        summaries = _summarize_chunks(llm, chunks, callbacks, verbose)

    raise LLMContextLengthExceededError(
        "Summaries could not be compacted to fit the context window."
    )


def _build_summary_prompt(chunk: list[LLMMessage]) -> list[LLMMessage]:
    return [
        format_message_for_llm(
            I18N_DEFAULT.slice("summarizer_system_message"), "system"
        ),
        format_message_for_llm(
            I18N_DEFAULT.slice("summarize_instruction").format(
                conversation=messages_to_text(chunk)
            )
        ),
    ]


def _chunk_messages(
    llm: Any, messages: list[LLMMessage], max_tokens: int
) -> list[list[LLMMessage]]:
    """Greedily pack rendered summary requests, splitting only oversized messages."""
    if _summary_request_tokens(llm, []) > max_tokens:
        raise LLMContextLengthExceededError(
            "Summarization prompt exceeds the context window before conversation history is added."
        )
    chunks: list[list[LLMMessage]] = []
    current: list[LLMMessage] = []
    for unit in _message_units(messages):
        if _summary_request_tokens(llm, [*current, *unit]) > max_tokens:
            if current:
                chunks.append(current)
                current = []
            if _summary_request_tokens(llm, unit) <= max_tokens:
                current.extend(unit)
                continue
            for message in unit:
                for part in _split_message_to_fit(llm, message, max_tokens):
                    if _summary_request_tokens(llm, [*current, part]) > max_tokens:
                        if not current:
                            raise LLMContextLengthExceededError(
                                "Summarization message could not be split to fit the context window."
                            )
                        chunks.append(current)
                        current = []
                    current.append(part)
            continue

        current.extend(unit)

    if current:
        chunks.append(current)
    return chunks


def _message_units(messages: list[LLMMessage]) -> list[list[LLMMessage]]:
    """Keep an assistant tool call with its following tool results when possible."""
    units: list[list[LLMMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "system":
            index += 1
            continue
        unit = [message]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                unit.append(messages[index])
                index += 1
            units.append(unit)
            continue
        units.append(unit)
        index += 1
    return units


def _split_message_to_fit(
    llm: Any, message: LLMMessage, max_tokens: int
) -> list[LLMMessage]:
    """Split an oversized message using the LLM's request-token estimator."""
    if _summary_request_tokens(llm, [message]) <= max_tokens:
        return [message]

    text = message_content_text(message)
    if not text:
        raise LLMContextLengthExceededError(
            "Summarization message has no text that can be split to fit the context window."
        )

    parts: list[LLMMessage] = []
    remainder = text
    while remainder:
        low, high, best = 1, len(remainder), 0
        while low <= high:
            length = (low + high) // 2
            candidate = cast("LLMMessage", {**message, "content": remainder[:length]})
            if _summary_request_tokens(llm, [candidate]) <= max_tokens:
                best = length
                low = length + 1
            else:
                high = length - 1

        if best == 0:
            raise LLMContextLengthExceededError(
                "Summarization message could not be split to fit the context window."
            )

        parts.append(cast("LLMMessage", {**message, "content": remainder[:best]}))
        remainder = remainder[best:]

    return parts


def _summary_request_tokens(llm: Any, messages: list[LLMMessage]) -> int:
    """Count the exact request used for a summary call through the active LLM."""
    return int(llm.get_input_tokens_count(_build_summary_prompt(messages)))
