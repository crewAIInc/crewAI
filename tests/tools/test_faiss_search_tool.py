import numpy as np
import pytest

from crewai.tools import FAISSSearchTool

def test_faiss_search_tool_initialization():
    tool = FAISSSearchTool()
    assert tool.name == "FAISS Search Tool"
    assert tool.dimension == 384

def test_faiss_search_with_texts():
    tool = FAISSSearchTool()
    texts = [
        "The quick brown fox",
        "jumps over the lazy dog",
        "A completely different text"
    ]
    tool.add_texts(texts)
    
    results = tool.run(
        query="quick fox",
        k=2,
        score_threshold=0.5
    )
    
    assert len(results) > 0
    assert isinstance(results[0]["text"], str)
    assert isinstance(results[0]["score"], float)

def test_faiss_search_threshold_filtering():
    tool = FAISSSearchTool()
    texts = ["Text A", "Text B", "Text C"]
    tool.add_texts(texts)
    
    results = tool.run(
        query="Something completely different",
        score_threshold=0.99  # High threshold
    )
    
    assert len(results) == 0  # No results above threshold

def test_invalid_index_type():
    with pytest.raises(ValueError):
        FAISSSearchTool(index_type="INVALID")
