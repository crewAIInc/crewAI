from tempfile import TemporaryDirectory
from typing import cast
from pathlib import Path


from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter
from crewai_tools.tools.rag.rag_tool import RagTool


def test_rag_tool_initialization():
    """Test that RagTool initializes with CrewAI adapter by default."""
    class MyTool(RagTool):
        pass

    tool = MyTool()
    assert tool.adapter is not None
    assert isinstance(tool.adapter, CrewAIRagAdapter)
    
    adapter = cast(CrewAIRagAdapter, tool.adapter)
    assert adapter.collection_name == "rag_tool_collection"
    assert adapter._client is not None


def test_rag_tool_add_and_query():
    """Test adding content and querying with RagTool."""
    class MyTool(RagTool):
        pass
    
    tool = MyTool()
    
    tool.add("The sky is blue on a clear day.")
    tool.add("Machine learning is a subset of artificial intelligence.")
    
    result = tool._run(query="What color is the sky?")
    assert "Relevant Content:" in result
    
    result = tool._run(query="Tell me about machine learning")
    assert "Relevant Content:" in result


def test_rag_tool_with_file():
    """Test RagTool with file content."""
    with TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("Python is a programming language known for its simplicity.")
        
        class MyTool(RagTool):
            pass
        
        tool = MyTool()
        tool.add(str(test_file))
        
        result = tool._run(query="What is Python?")
        assert "Relevant Content:" in result
