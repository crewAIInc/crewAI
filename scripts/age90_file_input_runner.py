# ruff: noqa: T201
"""Manual runner for AGE-90 PDF input handling.

Usage examples:
    uv run python scripts/age90_file_input_runner.py
    uv run python scripts/age90_file_input_runner.py --mode fallback
    uv run python scripts/age90_file_input_runner.py --mode payload --pdf ./sample_story.pdf
    uv run python scripts/age90_file_input_runner.py --mode kickoff --pdf ./sample_story.pdf
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from crewai_files import PDFFile, format_multimodal_content, get_supported_content_types


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "lib" / "crewai-files" / "tests" / "fixtures" / "agents.pdf"


def _content_summary(block: dict[str, Any]) -> dict[str, str]:
    """Return a compact, non-base64 summary of a content block."""
    summary: dict[str, str] = {"type": str(block.get("type"))}
    for key in ("file_id", "file_url", "filename", "image_url"):
        if key in block:
            value = str(block[key])
            summary[key] = value[:100] + ("..." if len(value) > 100 else "")
    if "file_data" in block:
        value = str(block["file_data"])
        summary["file_data"] = value[:80] + f"... ({len(value)} chars)"
    return summary


def _sanitize_payload(value: Any) -> Any:
    """Shorten large fields before printing API payloads."""
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "file_data" and isinstance(item, str):
                sanitized[key] = item[:100] + f"... ({len(item)} chars)"
            else:
                sanitized[str(key)] = _sanitize_payload(item)
        return sanitized

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_sanitize_payload(item) for item in value]

    return value


def inspect_native_path(pdf_path: Path, provider: str, api: str | None) -> None:
    """Show whether the PDF is treated as a native multimodal input."""
    pdf = PDFFile(source=str(pdf_path))
    supported_types = get_supported_content_types(provider, api=api)
    blocks = format_multimodal_content(
        {"document": pdf},
        provider=provider,
        api=api,
        text="Summarize this PDF.",
    )

    print("\n== Native File Formatting ==")
    print(f"PDF: {pdf_path}")
    print(f"Provider/API: {provider} / {api or 'default'}")
    print(f"Supported content types: {supported_types}")
    print(f"Content block count: {len(blocks)}")
    for index, block in enumerate(blocks, start=1):
        print(f"  {index}. {_content_summary(block)}")

    has_pdf_block = any(block.get("type") == "input_file" for block in blocks)
    print(f"PDF native input_file block: {'YES' if has_pdf_block else 'NO'}")


def inspect_fallback_tool(pdf_path: Path) -> None:
    """Show what read_file returns if a PDF falls back to the tool path."""
    from crewai.tools.agent_tools.read_file_tool import ReadFileTool

    tool = ReadFileTool()
    tool.set_files({"document": PDFFile(source=str(pdf_path))})
    result = tool._run("document")

    print("\n== read_file Fallback ==")
    print(f"Returned {len(result)} chars")
    print(f"Contains Base64 marker: {'YES' if 'Base64:' in result else 'NO'}")
    print("\nPreview:")
    print(result[:1200])
    if len(result) > 1200:
        print("...")


def run_crew_kickoff(
    pdf_path: Path,
    model: str,
    api: str | None,
    prompt: str,
    *,
    payload_only: bool = False,
) -> None:
    """Run a real Crew kickoff against the supplied model."""
    from crewai import LLM, Agent, Crew, Task

    if model.startswith("openai/") and not os.getenv("OPENAI_API_KEY") and not payload_only:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running --mode kickoff."
        )

    kwargs: dict[str, Any] = {"model": model, "temperature": 0}
    if api:
        kwargs["api"] = api

    llm = LLM(**kwargs)
    agent = Agent(
        role="PDF Analyst",
        goal="Read the provided PDF and answer accurately from its contents",
        backstory="You inspect uploaded files carefully and avoid guessing.",
        llm=llm,
        verbose=True,
    )
    task = Task(
        description=prompt,
        expected_output="A concise answer grounded in the uploaded PDF.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=True)

    print("\n== Crew Kickoff ==")
    print(f"Model/API: {model} / {api or 'default'}")
    print(f"PDF: {pdf_path}")

    context = nullcontext()
    if payload_only:
        from crewai.llms.providers.openai.completion import OpenAICompletion

        def print_payload_and_stop(
            self: OpenAICompletion,
            params: dict[str, Any],
            *_args: Any,
            **_kwargs: Any,
        ) -> str:
            print("\n== Sanitized Responses Payload ==")
            print(_sanitize_payload(params))
            return "Payload debug complete."

        context = patch.object(
            OpenAICompletion,
            "_handle_responses",
            print_payload_and_stop,
        )

    with context:
        result = crew.kickoff(input_files={"document": PDFFile(source=str(pdf_path))})

    print("\n== Final Output ==")
    print(result.raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inspect", "fallback", "payload", "kickoff", "all"),
        default="inspect",
        help="What to run. 'inspect', 'fallback', and 'payload' do not call an LLM.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="PDF file to test.",
    )
    parser.add_argument(
        "--provider",
        default="gpt-4o-mini",
        help="Provider/model string for file formatting inspection.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="CrewAI model for real kickoff mode.",
    )
    parser.add_argument(
        "--api",
        default="responses",
        help="API variant. Use '' to omit.",
    )
    parser.add_argument(
        "--prompt",
        default="Summarize the uploaded PDF in 3 bullet points. Do not guess.",
        help="Task prompt for kickoff mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    api = args.api or None

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    if args.mode in ("inspect", "all"):
        inspect_native_path(pdf_path, args.provider, api)
    if args.mode in ("fallback", "all"):
        inspect_fallback_tool(pdf_path)
    if args.mode == "payload":
        run_crew_kickoff(pdf_path, args.model, api, args.prompt, payload_only=True)
    if args.mode in ("kickoff", "all"):
        run_crew_kickoff(
            pdf_path,
            args.model,
            api,
            args.prompt,
            payload_only=args.mode == "all",
        )


if __name__ == "__main__":
    main()
