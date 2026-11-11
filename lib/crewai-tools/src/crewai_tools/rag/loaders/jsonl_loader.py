import json
from typing import Any, NoReturn

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.loaders.utils import load_from_url
from crewai_tools.rag.source_content import SourceContent


class JSONLLoader(BaseLoader):
    """Load a UTF-8 file, URL, or inline JSONL into one document.

    Empty input produces empty content with a record count of zero.
    Invalid records raise ValueError with their 1-based line number.
    """

    def load(self, content: SourceContent, **kwargs: Any) -> LoaderResult:
        """Validate JSONL records and return their text, count, and source identity."""
        source_ref = content.source_ref
        text = content.source
        if content.is_url():
            text = load_from_url(
                source_ref,
                kwargs,
                accept_header="application/jsonl, application/x-ndjson, text/plain",
                loader_name="JSONLLoader",
            )
        elif content.path_exists():
            with open(source_ref, encoding="utf-8", newline="") as file:
                text = file.read()

        lines = text.split("\n")
        if lines[-1] == "":
            lines.pop()

        records = []
        for line_number, line in enumerate(lines, start=1):
            try:
                json.loads(line, parse_constant=self._reject_constant)
            except ValueError as error:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {error}"
                ) from error
            records.append(line.strip(" \t\r"))

        text = "\n".join(records)
        return LoaderResult(
            content=text,
            source=source_ref,
            metadata={"format": "jsonl", "record_count": len(records)},
            doc_id=self.generate_doc_id(source_ref=source_ref, content=text),
        )

    @staticmethod
    def _reject_constant(value: str) -> NoReturn:
        """Reject NaN and infinity tokens, which are not valid JSON values."""
        raise ValueError(f"Invalid JSON constant: {value}")
