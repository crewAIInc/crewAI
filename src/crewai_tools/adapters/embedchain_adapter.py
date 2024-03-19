from typing import Any

from embedchain import App

from crewai_tools.tools.rag.rag_tool import Adapter


class EmbedchainAdapter(Adapter):
    embedchain_app: App
    summarize: bool = False

    def query(self, question: str) -> str:
        result, sources = self.embedchain_app.query(
            question, citations=True, dry_run=(not self.summarize)
        )
        if self.summarize:
            return result
        return "\n\n".join([source[0] for source in sources])

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.embedchain_app.add(*args, **kwargs)
