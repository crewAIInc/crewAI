from typing import Any, Optional

from embedchain import App

from crewai_tools.tools.rag.rag_tool import Adapter


class PDFEmbedchainAdapter(Adapter):
    embedchain_app: App
    summarize: bool = False
    src: Optional[str] = None

    def query(self, question: str) -> str:
        where = (
            {"app_id": self.embedchain_app.config.id, "source": self.src}
            if self.src
            else None
        )
        result, sources = self.embedchain_app.query(
            question, citations=True, dry_run=(not self.summarize), where=where
        )
        if self.summarize:
            return result
        return "\n\n".join([source[0] for source in sources])

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.src = args[0] if args else None
        self.embedchain_app.add(*args, **kwargs)
