from crewai_tools.tools.rag.rag_tool import Adapter, RagTool


class MockAdapter(Adapter):
    answer: str

    def query(self, question: str) -> str:
        return self.answer


def test_rag_tool():
    adapter = MockAdapter(answer="42")
    rag_tool = RagTool(adapter=adapter)

    assert rag_tool.name == "Knowledge base"
    assert (
        rag_tool.description == "A knowledge base that can be used to answer questions."
    )
    assert (
        rag_tool.run("What is the answer to life, the universe and everything?") == "42"
    )
