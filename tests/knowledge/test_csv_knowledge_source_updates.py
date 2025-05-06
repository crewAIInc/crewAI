import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


@patch('crewai.knowledge.storage.knowledge_storage.KnowledgeStorage.search')
@patch('crewai.knowledge.source.csv_knowledge_source.CSVKnowledgeSource.add')
def test_csv_knowledge_source_updates(mock_add, mock_search, tmpdir):
    """Test that CSVKnowledgeSource properly detects and loads updates to CSV files."""
    mock_search.side_effect = [
        [{"context": "name,age,city\nJohn,30,New York\nAlice,25,San Francisco\nBob,28,Chicago"}],
        [{"context": "name,age,city\nJohn,30,Boston\nAlice,25,San Francisco\nBob,28,Chicago\nEve,22,Miami"}],
        [{"context": "name,age,city\nJohn,30,Boston\nAlice,25,San Francisco\nBob,28,Chicago\nEve,22,Miami"}]
    ]
    
    csv_path = tmpdir / "test_updates.csv"
    
    initial_csv_content = [
        ["name", "age", "city"],
        ["John", "30", "New York"],
        ["Alice", "25", "San Francisco"],
        ["Bob", "28", "Chicago"],
    ]
    
    with open(csv_path, "w") as f:
        for row in initial_csv_content:
            f.write(",".join(row) + "\n")
    
    csv_source = CSVKnowledgeSource(file_paths=[csv_path])
    
    original_files_have_changed = csv_source.files_have_changed
    files_changed_called = [False]
    
    def spy_files_have_changed():
        files_changed_called[0] = True
        return original_files_have_changed()
    
    csv_source.files_have_changed = spy_files_have_changed
    
    knowledge = Knowledge(sources=[csv_source], collection_name="test_updates")
    
    assert hasattr(knowledge, '_check_and_reload_sources'), "Knowledge class is missing _check_and_reload_sources method"
    
    initial_results = knowledge.query(["John"])
    assert any("John" in result["context"] for result in initial_results)
    assert any("New York" in result["context"] for result in initial_results)
    
    mock_add.reset_mock()
    files_changed_called[0] = False
    
    updated_csv_content = [
        ["name", "age", "city"],
        ["John", "30", "Boston"],  # Changed city
        ["Alice", "25", "San Francisco"],
        ["Bob", "28", "Chicago"],
        ["Eve", "22", "Miami"],  # Added new person
    ]
    
    time.sleep(1)
    
    with open(csv_path, "w") as f:
        for row in updated_csv_content:
            f.write(",".join(row) + "\n")
    
    updated_results = knowledge.query(["John"])
    
    assert files_changed_called[0], "files_have_changed method was not called during query"
    
    assert mock_add.called, "add method was not called to reload the data"
    
    assert any("John" in result["context"] for result in updated_results)
    assert any("Boston" in result["context"] for result in updated_results)
    assert not any("New York" in result["context"] for result in updated_results)
    
    new_results = knowledge.query(["Eve"])
    assert any("Eve" in result["context"] for result in new_results)
    assert any("Miami" in result["context"] for result in new_results)
