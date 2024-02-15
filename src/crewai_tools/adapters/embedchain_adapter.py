from embedchain import App

from crewai_tools.tools.rag.rag_tool import Adapter


class EmbedchainAdapter(Adapter):
    embedchain_app: App
    dry_run: bool = False

    def query(self, question: str) -> str:
        result = self.embedchain_app.query(question, dry_run=self.dry_run)
        if result is list:
            return "\n".join(result)
        return str(result)
