from typing import Any

from crewai_tools.tools.rag.rag_tool import Adapter

try:
    from embedchain import App
    EMBEDCHAIN_AVAILABLE = True
except ImportError:
    EMBEDCHAIN_AVAILABLE = False


class EmbedchainAdapter(Adapter):
    embedchain_app: Any  # Will be App when embedchain is available
    summarize: bool = False

    def __init__(self, **data):
        if not EMBEDCHAIN_AVAILABLE:
            raise ImportError("embedchain is not installed. Please install it with `pip install crewai-tools[embedchain]`")
        super().__init__(**data)

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
