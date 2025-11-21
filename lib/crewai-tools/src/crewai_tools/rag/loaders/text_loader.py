from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class TextFileLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        source_ref = source_content.source_ref
        if not source_content.path_exists():
            raise FileNotFoundError(
                f"The following file does not exist: {source_content.source}"
            )

        with open(source_content.source, encoding="utf-8") as file:
            content = file.read()

        return LoaderResult(
            content=content,
            source=source_ref,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=content),
        )


class TextLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        return LoaderResult(
            content=source_content.source,
            source=source_content.source_ref,
            doc_id=self.generate_doc_id(content=source_content.source),
        )
