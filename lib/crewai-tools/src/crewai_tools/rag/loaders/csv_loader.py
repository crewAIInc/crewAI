import csv
from io import StringIO

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.loaders.utils import load_from_url
from crewai_tools.rag.source_content import SourceContent


class CSVLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        source_ref = source_content.source_ref

        content_str = source_content.source
        if source_content.is_url():
            content_str = load_from_url(
                content_str,
                kwargs,
                accept_header="text/csv, application/csv, text/plain",
                loader_name="CSVLoader",
            )
        elif source_content.path_exists():
            content_str = self._load_from_file(content_str)

        return self._parse_csv(content_str, source_ref)

    @staticmethod
    def _load_from_file(path: str) -> str:
        with open(path, encoding="utf-8") as file:
            return file.read()

    def _parse_csv(self, content: str, source_ref: str) -> LoaderResult:
        try:
            csv_reader = csv.DictReader(StringIO(content))

            text_parts = []
            headers = csv_reader.fieldnames

            if headers:
                text_parts.append("Headers: " + " | ".join(headers))
                text_parts.append("-" * 50)

                for row_num, row in enumerate(csv_reader, 1):
                    row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                    text_parts.append(f"Row {row_num}: {row_text}")

            text = "\n".join(text_parts)

            metadata = {
                "format": "csv",
                "columns": headers,
                "rows": len(text_parts) - 2 if headers else 0,
            }

        except Exception as e:
            text = content
            metadata = {"format": "csv", "parse_error": str(e)}

        return LoaderResult(
            content=text,
            source=source_ref,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=text),
        )
