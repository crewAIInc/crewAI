import json
from typing import Any

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class JSONLLoader(BaseLoader):
    """Load a UTF-8 file or inline JSONL into one document.

    Empty input produces empty content with a record count of zero.
    """

    def load(self, content: SourceContent, **kwargs: Any) -> LoaderResult:
        source_ref = content.source_ref
        text = content.source
        if content.path_exists():
            with open(source_ref, encoding="utf-8", newline="") as file:
                text = file.read()

        lines = text.split("\n")
        if lines[-1] == "":
            lines.pop()

        records = []
        for line in lines:
            json.loads(line)
            records.append(line.strip(" \t\r"))

        text = "\n".join(records)
        return LoaderResult(
            content=text,
            source=source_ref,
            metadata={"format": "jsonl", "record_count": len(records)},
            doc_id=self.generate_doc_id(source_ref=source_ref, content=text),
        )
