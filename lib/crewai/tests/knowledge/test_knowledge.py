"""Test Knowledge creation and querying functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


@pytest.fixture(autouse=True)
def mock_vector_db():
    """Mock vector database operations."""
    with patch("crewai.knowledge.storage.knowledge_storage.KnowledgeStorage") as mock:
        # Mock the query method to return a predefined response
        instance = mock.return_value
        instance.query.return_value = [
            {
                "content": "Brandon's favorite color is blue and he likes Mexican food.",
                "score": 0.9,
            }
        ]
        instance.reset.return_value = None
        yield instance


@pytest.fixture(autouse=True)
def reset_knowledge_storage(mock_vector_db):
    """Fixture to reset knowledge storage before each test."""
    yield


def test_single_short_string(mock_vector_db):
    # Create a knowledge base with a single short string
    content = "Brandon's favorite color is blue and he likes Mexican food."
    string_source = StringKnowledgeSource(
        content=content, metadata={"preference": "personal"}
    )
    mock_vector_db.sources = [string_source]
    mock_vector_db.query.return_value = [{"content": content, "score": 0.9}]
    # Perform a query
    query = "What is Brandon's favorite color?"
    results = mock_vector_db.query(query)

    # Assert that the results contain the expected information
    assert any("blue" in result["content"].lower() for result in results)
    # Verify the mock was called
    mock_vector_db.query.assert_called_once()


# @pytest.mark.vcr()
def test_single_2k_character_string(mock_vector_db):
    # Create a 2k character string with various facts about Brandon
    content = (
        "Brandon is a software engineer who lives in San Francisco. "
        "He enjoys hiking and often visits the trails in the Bay Area. "
        "Brandon has a pet dog named Max, who is a golden retriever. "
        "He loves reading science fiction books, and his favorite author is Isaac Asimov. "
        "Brandon's favorite movie is Inception, and he enjoys watching it with his friends. "
        "He is also a fan of Mexican cuisine, especially tacos and burritos. "
        "Brandon plays the guitar and often performs at local open mic nights. "
        "He is learning French and plans to visit Paris next year. "
        "Brandon is passionate about technology and often attends tech meetups in the city. "
        "He is also interested in AI and machine learning, and he is currently working on a project related to natural language processing. "
        "Brandon's favorite color is blue, and he often wears blue shirts. "
        "He enjoys cooking and often tries new recipes on weekends. "
        "Brandon is a morning person and likes to start his day with a run in the park. "
        "He is also a coffee enthusiast and enjoys trying different coffee blends. "
        "Brandon is a member of a local book club and enjoys discussing books with fellow members. "
        "He is also a fan of board games and often hosts game nights at his place. "
        "Brandon is an advocate for environmental conservation and volunteers for local clean-up drives. "
        "He is also a mentor for aspiring software developers and enjoys sharing his knowledge with others. "
        "Brandon's favorite sport is basketball, and he often plays with his friends on weekends. "
        "He is also a fan of the Golden State Warriors and enjoys watching their games. "
    )
    string_source = StringKnowledgeSource(
        content=content, metadata={"preference": "personal"}
    )
    mock_vector_db.sources = [string_source]
    mock_vector_db.query.return_value = [{"content": content, "score": 0.9}]

    # Perform a query
    query = "What is Brandon's favorite movie?"
    results = mock_vector_db.query(query)

    # Assert that the results contain the expected information
    assert any("inception" in result["content"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_multiple_short_strings(mock_vector_db):
    # Create multiple short string sources
    contents = [
        "Brandon loves hiking.",
        "Brandon has a dog named Max.",
        "Brandon enjoys painting landscapes.",
    ]
    string_sources = [
        StringKnowledgeSource(content=content, metadata={"preference": "personal"})
        for content in contents
    ]

    # Mock the vector db query response
    mock_vector_db.query.return_value = [
        {"content": "Brandon has a dog named Max.", "score": 0.9}
    ]

    mock_vector_db.sources = string_sources

    # Perform a query
    query = "What is the name of Brandon's pet?"
    results = mock_vector_db.query(query)

    # Assert that the correct information is retrieved
    assert any("max" in result["content"].lower() for result in results)
    # Verify the mock was called
    mock_vector_db.query.assert_called_once()


def test_multiple_2k_character_strings(mock_vector_db):
    # Create multiple 2k character strings with various facts about Brandon
    contents = [
        (
            "Brandon is a software engineer who lives in San Francisco. "
            "He enjoys hiking and often visits the trails in the Bay Area. "
            "Brandon has a pet dog named Max, who is a golden retriever. "
            "He loves reading science fiction books, and his favorite author is Isaac Asimov. "
            "Brandon's favorite movie is Inception, and he enjoys watching it with his friends. "
            "He is also a fan of Mexican cuisine, especially tacos and burritos. "
            "Brandon plays the guitar and often performs at local open mic nights. "
            "He is learning French and plans to visit Paris next year. "
            "Brandon is passionate about technology and often attends tech meetups in the city. "
            "He is also interested in AI and machine learning, and he is currently working on a project related to natural language processing. "
            "Brandon's favorite color is blue, and he often wears blue shirts. "
            "He enjoys cooking and often tries new recipes on weekends. "
            "Brandon is a morning person and likes to start his day with a run in the park. "
            "He is also a coffee enthusiast and enjoys trying different coffee blends. "
            "Brandon is a member of a local book club and enjoys discussing books with fellow members. "
            "He is also a fan of board games and often hosts game nights at his place. "
            "Brandon is an advocate for environmental conservation and volunteers for local clean-up drives. "
            "He is also a mentor for aspiring software developers and enjoys sharing his knowledge with others. "
            "Brandon's favorite sport is basketball, and he often plays with his friends on weekends. "
            "He is also a fan of the Golden State Warriors and enjoys watching their games. "
        )
        * 2,  # Repeat to ensure it's 2k characters
        (
            "Brandon loves traveling and has visited over 20 countries. "
            "He is fluent in Spanish and often practices with his friends. "
            "Brandon's favorite city is Barcelona, where he enjoys the architecture and culture. "
            "He is a foodie and loves trying new cuisines, with a particular fondness for sushi. "
            "Brandon is an avid cyclist and participates in local cycling events. "
            "He is also a photographer and enjoys capturing landscapes and cityscapes. "
            "Brandon is a tech enthusiast and follows the latest trends in gadgets and software. "
            "He is also a fan of virtual reality and owns a VR headset. "
            "Brandon's favorite book is 'The Hitchhiker's Guide to the Galaxy'. "
            "He enjoys watching documentaries and learning about history and science. "
            "Brandon is a coffee lover and has a collection of coffee mugs from different countries. "
            "He is also a fan of jazz music and often attends live performances. "
            "Brandon is a member of a local running club and participates in marathons. "
            "He is also a volunteer at a local animal shelter and helps with dog walking. "
            "Brandon's favorite holiday is Christmas, and he enjoys decorating his home. "
            "He is also a fan of classic movies and has a collection of DVDs. "
            "Brandon is a mentor for young professionals and enjoys giving career advice. "
            "He is also a fan of puzzles and enjoys solving them in his free time. "
            "Brandon's favorite sport is soccer, and he often plays with his friends. "
            "He is also a fan of FC Barcelona and enjoys watching their matches. "
        )
        * 2,  # Repeat to ensure it's 2k characters
    ]
    string_sources = [
        StringKnowledgeSource(content=content, metadata={"preference": "personal"})
        for content in contents
    ]

    mock_vector_db.sources = string_sources
    mock_vector_db.query.return_value = [{"content": contents[1], "score": 0.9}]

    # Perform a query
    query = "What is Brandon's favorite book?"
    results = mock_vector_db.query(query)

    # Assert that the correct information is retrieved
    assert any(
        "the hitchhiker's guide to the galaxy" in result["content"].lower()
        for result in results
    )
    mock_vector_db.query.assert_called_once()


def test_single_short_file(mock_vector_db, tmpdir):
    # Create a single short text file
    content = "Brandon's favorite sport is basketball."
    file_path = Path(tmpdir.join("short_file.txt"))
    with open(file_path, "w") as f:
        f.write(content)

    file_source = TextFileKnowledgeSource(
        file_paths=[file_path], metadata={"preference": "personal"}
    )
    mock_vector_db.sources = [file_source]
    mock_vector_db.query.return_value = [{"content": content, "score": 0.9}]
    # Perform a query
    query = "What sport does Brandon like?"
    results = mock_vector_db.query(query)

    # Assert that the results contain the expected information
    assert any("basketball" in result["content"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_single_2k_character_file(mock_vector_db, tmpdir):
    # Create a single 2k character text file with various facts about Brandon
    content = (
        "Brandon is a software engineer who lives in San Francisco. "
        "He enjoys hiking and often visits the trails in the Bay Area. "
        "Brandon has a pet dog named Max, who is a golden retriever. "
        "He loves reading science fiction books, and his favorite author is Isaac Asimov. "
        "Brandon's favorite movie is Inception, and he enjoys watching it with his friends. "
        "He is also a fan of Mexican cuisine, especially tacos and burritos. "
        "Brandon plays the guitar and often performs at local open mic nights. "
        "He is learning French and plans to visit Paris next year. "
        "Brandon is passionate about technology and often attends tech meetups in the city. "
        "He is also interested in AI and machine learning, and he is currently working on a project related to natural language processing. "
        "Brandon's favorite color is blue, and he often wears blue shirts. "
        "He enjoys cooking and often tries new recipes on weekends. "
        "Brandon is a morning person and likes to start his day with a run in the park. "
        "He is also a coffee enthusiast and enjoys trying different coffee blends. "
        "Brandon is a member of a local book club and enjoys discussing books with fellow members. "
        "He is also a fan of board games and often hosts game nights at his place. "
        "Brandon is an advocate for environmental conservation and volunteers for local clean-up drives. "
        "He is also a mentor for aspiring software developers and enjoys sharing his knowledge with others. "
        "Brandon's favorite sport is basketball, and he often plays with his friends on weekends. "
        "He is also a fan of the Golden State Warriors and enjoys watching their games. "
    ) * 2  # Repeat to ensure it's 2k characters
    file_path = Path(tmpdir.join("long_file.txt"))
    with open(file_path, "w") as f:
        f.write(content)

    file_source = TextFileKnowledgeSource(
        file_paths=[file_path], metadata={"preference": "personal"}
    )
    mock_vector_db.sources = [file_source]
    mock_vector_db.query.return_value = [{"content": content, "score": 0.9}]
    # Perform a query
    query = "What is Brandon's favorite movie?"
    results = mock_vector_db.query(query)

    # Assert that the results contain the expected information
    assert any("inception" in result["content"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_multiple_short_files(mock_vector_db, tmpdir):
    # Create multiple short text files
    contents = [
        {
            "content": "Brandon works as a software engineer.",
            "metadata": {"category": "profession", "source": "occupation"},
        },
        {
            "content": "Brandon lives in New York.",
            "metadata": {"category": "city", "source": "personal"},
        },
        {
            "content": "Brandon enjoys cooking Italian food.",
            "metadata": {"category": "hobby", "source": "personal"},
        },
    ]
    file_paths = []
    for i, item in enumerate(contents):
        file_path = Path(tmpdir.join(f"file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(item["content"])
        file_paths.append((file_path, item["metadata"]))

    file_sources = [
        TextFileKnowledgeSource(file_paths=[path], metadata=metadata)
        for path, metadata in file_paths
    ]
    mock_vector_db.sources = file_sources
    mock_vector_db.query.return_value = [
        {"content": "Brandon lives in New York.", "score": 0.9}
    ]
    # Perform a query
    query = "What city does he reside in?"
    results = mock_vector_db.query(query)
    # Assert that the correct information is retrieved
    assert any("new york" in result["content"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_multiple_2k_character_files(mock_vector_db, tmpdir):
    # Create multiple 2k character text files with various facts about Brandon
    contents = [
        (
            "Brandon loves traveling and has visited over 20 countries. "
            "He is fluent in Spanish and often practices with his friends. "
            "Brandon's favorite city is Barcelona, where he enjoys the architecture and culture. "
            "He is a foodie and loves trying new cuisines, with a particular fondness for sushi. "
            "Brandon is an avid cyclist and participates in local cycling events. "
            "He is also a photographer and enjoys capturing landscapes and cityscapes. "
            "Brandon is a tech enthusiast and follows the latest trends in gadgets and software. "
            "He is also a fan of virtual reality and owns a VR headset. "
            "Brandon's favorite book is 'The Hitchhiker's Guide to the Galaxy'. "
            "He enjoys watching documentaries and learning about history and science. "
            "Brandon is a coffee lover and has a collection of coffee mugs from different countries. "
            "He is also a fan of jazz music and often attends live performances. "
            "Brandon is a member of a local running club and participates in marathons. "
            "He is also a volunteer at a local animal shelter and helps with dog walking. "
            "Brandon's favorite holiday is Christmas, and he enjoys decorating his home. "
            "He is also a fan of classic movies and has a collection of DVDs. "
            "Brandon is a mentor for young professionals and enjoys giving career advice. "
            "He is also a fan of puzzles and enjoys solving them in his free time. "
            "Brandon's favorite sport is soccer, and he often plays with his friends. "
            "He is also a fan of FC Barcelona and enjoys watching their matches. "
        )
        * 2,  # Repeat to ensure it's 2k characters
        (
            "Brandon is a software engineer who lives in San Francisco. "
            "He enjoys hiking and often visits the trails in the Bay Area. "
            "Brandon has a pet dog named Max, who is a golden retriever. "
            "He loves reading science fiction books, and his favorite author is Isaac Asimov. "
            "Brandon's favorite movie is Inception, and he enjoys watching it with his friends. "
            "He is also a fan of Mexican cuisine, especially tacos and burritos. "
            "Brandon plays the guitar and often performs at local open mic nights. "
            "He is learning French and plans to visit Paris next year. "
            "Brandon is passionate about technology and often attends tech meetups in the city. "
            "He is also interested in AI and machine learning, and he is currently working on a project related to natural language processing. "
            "Brandon's favorite color is blue, and he often wears blue shirts. "
            "He enjoys cooking and often tries new recipes on weekends. "
            "Brandon is a morning person and likes to start his day with a run in the park. "
            "He is also a coffee enthusiast and enjoys trying different coffee blends. "
            "Brandon is a member of a local book club and enjoys discussing books with fellow members. "
            "He is also a fan of board games and often hosts game nights at his place. "
            "Brandon is an advocate for environmental conservation and volunteers for local clean-up drives. "
            "He is also a mentor for aspiring software developers and enjoys sharing his knowledge with others. "
            "Brandon's favorite sport is basketball, and he often plays with his friends on weekends. "
            "He is also a fan of the Golden State Warriors and enjoys watching their games. "
        )
        * 2,  # Repeat to ensure it's 2k characters
    ]
    file_paths = []
    for i, content in enumerate(contents):
        file_path = Path(tmpdir.join(f"long_file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(content)
        file_paths.append(file_path)

    file_sources = [
        TextFileKnowledgeSource(file_paths=[path], metadata={"preference": "personal"})
        for path in file_paths
    ]
    mock_vector_db.sources = file_sources
    mock_vector_db.query.return_value = [
        {
            "content": "Brandon's favorite book is 'The Hitchhiker's Guide to the Galaxy'.",
            "score": 0.9,
        }
    ]
    # Perform a query
    query = "What is Brandon's favorite book?"
    results = mock_vector_db.query(query)

    # Assert that the correct information is retrieved
    assert any(
        "the hitchhiker's guide to the galaxy" in result["content"].lower()
        for result in results
    )
    mock_vector_db.query.assert_called_once()


@pytest.mark.vcr()
def test_hybrid_string_and_files(mock_vector_db, tmpdir):
    # Create string sources
    string_contents = [
        "Brandon is learning French.",
        "Brandon visited Paris last summer.",
    ]
    string_sources = [
        StringKnowledgeSource(content=content, metadata={"preference": "personal"})
        for content in string_contents
    ]

    # Create file sources
    file_contents = [
        "Brandon prefers tea over coffee.",
        "Brandon's favorite book is 'The Alchemist'.",
    ]
    file_paths = []
    for i, content in enumerate(file_contents):
        file_path = Path(tmpdir.join(f"file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(content)
        file_paths.append(file_path)

    file_sources = [
        TextFileKnowledgeSource(file_paths=[path], metadata={"preference": "personal"})
        for path in file_paths
    ]

    # Combine string and file sources
    mock_vector_db.sources = string_sources + file_sources
    mock_vector_db.query.return_value = [{"content": file_contents[1], "score": 0.9}]

    # Perform a query
    query = "What is Brandon's favorite book?"
    results = mock_vector_db.query(query)

    # Assert that the correct information is retrieved
    assert any("the alchemist" in result["content"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_pdf_knowledge_source(mock_vector_db):
    # Get the directory of the current file
    current_dir = Path(__file__).parent
    # Construct the path to the PDF file
    pdf_path = current_dir / "crewai_quickstart.pdf"

    # Create a PDFKnowledgeSource
    pdf_source = PDFKnowledgeSource(
        file_paths=[pdf_path], metadata={"preference": "personal"}
    )
    mock_vector_db.sources = [pdf_source]
    mock_vector_db.query.return_value = [
        {"content": "crewai create crew latest-ai-development", "score": 0.9}
    ]

    # Perform a query
    query = "How do you create a crew?"
    results = mock_vector_db.query(query)

    # Assert that the correct information is retrieved
    assert any(
        "crewai create crew latest-ai-development" in result["content"].lower()
        for result in results
    )
    mock_vector_db.query.assert_called_once()


@pytest.mark.vcr()
def test_csv_knowledge_source(mock_vector_db, tmpdir):
    """Test CSVKnowledgeSource with a simple CSV file."""

    # Create a CSV file with sample data
    csv_content = [
        ["Name", "Age", "City"],
        ["Brandon", "30", "New York"],
        ["Alice", "25", "Los Angeles"],
        ["Bob", "35", "Chicago"],
    ]
    csv_path = Path(tmpdir.join("data.csv"))
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_content:
            f.write(",".join(row) + "\n")

    # Create a CSVKnowledgeSource
    csv_source = CSVKnowledgeSource(
        file_paths=[csv_path], metadata={"preference": "personal"}
    )
    mock_vector_db.sources = [csv_source]
    mock_vector_db.query.return_value = [
        {"content": "Brandon is 30 years old.", "score": 0.9}
    ]

    # Perform a query
    query = "How old is Brandon?"
    results = mock_vector_db.query(query)

    # Assert that the correct information is retrieved
    assert any("30" in result["content"] for result in results)
    mock_vector_db.query.assert_called_once()


def test_json_knowledge_source(mock_vector_db, tmpdir):
    """Test JSONKnowledgeSource with a simple JSON file."""

    # Create a JSON file with sample data
    json_data = {
        "people": [
            {"name": "Brandon", "age": 30, "city": "New York"},
            {"name": "Alice", "age": 25, "city": "Los Angeles"},
            {"name": "Bob", "age": 35, "city": "Chicago"},
        ]
    }
    json_path = Path(tmpdir.join("data.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        import json

        json.dump(json_data, f)

    # Create a JSONKnowledgeSource
    json_source = JSONKnowledgeSource(
        file_paths=[json_path], metadata={"preference": "personal"}
    )
    mock_vector_db.sources = [json_source]
    mock_vector_db.query.return_value = [
        {"content": "Alice lives in Los Angeles.", "score": 0.9}
    ]

    # Perform a query
    query = "Where does Alice reside?"
    results = mock_vector_db.query(query)

    # Assert that the correct information is retrieved
    assert any("los angeles" in result["content"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_excel_knowledge_source(mock_vector_db, tmpdir):
    """Test ExcelKnowledgeSource with a simple Excel file."""

    # Create an Excel file with sample data
    import pandas as pd  # type: ignore[import-untyped]

    excel_data = {
        "Name": ["Brandon", "Alice", "Bob"],
        "Age": [30, 25, 35],
        "City": ["New York", "Los Angeles", "Chicago"],
    }
    df = pd.DataFrame(excel_data)
    excel_path = Path(tmpdir.join("data.xlsx"))
    df.to_excel(excel_path, index=False)

    # Create an ExcelKnowledgeSource
    excel_source = ExcelKnowledgeSource(
        file_paths=[excel_path], metadata={"preference": "personal"}
    )
    mock_vector_db.sources = [excel_source]
    mock_vector_db.query.return_value = [
        {"content": "Brandon is 30 years old.", "score": 0.9}
    ]

    # Perform a query
    query = "What is Brandon's age?"
    results = mock_vector_db.query(query)

    # Assert that the correct information is retrieved
    assert any("30" in result["content"] for result in results)
    mock_vector_db.query.assert_called_once()


@pytest.mark.vcr
def test_docling_source(mock_vector_db):
    docling_source = CrewDoclingSource(
        file_paths=[
            "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        ],
    )
    mock_vector_db.sources = [docling_source]
    mock_vector_db.query.return_value = [
        {
            "content": "Reward hacking is a technique used to improve the performance of reinforcement learning agents.",
            "score": 0.9,
        }
    ]
    # Perform a query
    query = "What is reward hacking?"
    results = mock_vector_db.query(query)
    assert any("reward hacking" in result["content"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


@pytest.mark.vcr
def test_multiple_docling_sources() -> None:
    urls: list[Path | str] = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    ]
    docling_source = CrewDoclingSource(file_paths=urls)

    assert docling_source.file_paths == urls
    assert docling_source.content is not None


def test_file_path_validation():
    """Test file path validation for knowledge sources."""
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "crewai_quickstart.pdf"

    # Test valid single file_path
    source = PDFKnowledgeSource(file_path=pdf_path)
    assert source.safe_file_paths == [pdf_path]

    # Test valid file_paths list
    source = PDFKnowledgeSource(file_paths=[pdf_path])
    assert source.safe_file_paths == [pdf_path]

    # Test both file_path and file_paths provided (should use file_paths)
    source = PDFKnowledgeSource(file_path=pdf_path, file_paths=[pdf_path])
    assert source.safe_file_paths == [pdf_path]

    # Test neither file_path nor file_paths provided
    with pytest.raises(
        ValueError,
        match="file_path/file_paths must be a Path, str, or a list of these types",
    ):
        PDFKnowledgeSource()


def test_hash_based_id_generation_without_doc_id(mock_vector_db):
    """Test that documents without doc_id generate hash-based IDs. Duplicates are deduplicated before upsert."""
    import hashlib
    import json
    from crewai.rag.chromadb.utils import _prepare_documents_for_chromadb
    from crewai.rag.types import BaseRecord

    documents: list[BaseRecord] = [
        {"content": "First document content", "metadata": {"source": "test1", "category": "research"}},
        {"content": "Second document content", "metadata": {"source": "test2", "category": "research"}},
        {"content": "Third document content"},  # No metadata
    ]

    result = _prepare_documents_for_chromadb(documents)

    assert len(result.ids) == 3

    # Unique documents should get 64-character hex hashes (no suffix)
    for doc_id in result.ids:
        assert len(doc_id) == 64, f"ID should be 64 characters: {doc_id}"
        assert all(c in "0123456789abcdef" for c in doc_id), f"ID should be hex: {doc_id}"

    # Different documents should have different hashes
    assert result.ids[0] != result.ids[1] != result.ids[2]

    # Verify hashes match expected values
    expected_hash_1 = hashlib.sha256(
        f"First document content|{json.dumps({'category': 'research', 'source': 'test1'}, sort_keys=True)}".encode()
    ).hexdigest()
    assert result.ids[0] == expected_hash_1, "First document hash should match expected"

    expected_hash_3 = hashlib.sha256("Third document content".encode()).hexdigest()
    assert result.ids[2] == expected_hash_3, "Third document hash should match expected"

    # Test that duplicate documents are deduplicated (same ID, only one sent)
    duplicate_documents: list[BaseRecord] = [
        {"content": "Same content", "metadata": {"source": "test"}},
        {"content": "Same content", "metadata": {"source": "test"}},
        {"content": "Same content", "metadata": {"source": "test"}},
    ]
    duplicate_result = _prepare_documents_for_chromadb(duplicate_documents)
    # Duplicates should be deduplicated - only one ID should remain
    assert len(duplicate_result.ids) == 1, "Duplicate documents should be deduplicated"
    assert len(duplicate_result.ids[0]) == 64, "Deduplicated ID should be clean hash"
    # Verify it's the expected hash
    expected_hash = hashlib.sha256(
        f"Same content|{json.dumps({'source': 'test'}, sort_keys=True)}".encode()
    ).hexdigest()
    assert duplicate_result.ids[0] == expected_hash, "Deduplicated ID should match expected hash"


def test_hash_based_id_generation_with_doc_id_in_metadata(mock_vector_db):
    """Test that documents with doc_id in metadata use the doc_id directly, not hash-based."""
    from crewai.rag.chromadb.utils import _prepare_documents_for_chromadb
    from crewai.rag.types import BaseRecord

    documents_with_doc_id: list[BaseRecord] = [
        {"content": "First document", "metadata": {"doc_id": "custom-id-1", "source": "test1"}},
        {"content": "Second document", "metadata": {"doc_id": "custom-id-2"}},
    ]

    documents_without_doc_id: list[BaseRecord] = [
        {"content": "First document", "metadata": {"source": "test1"}},
        {"content": "Second document"},
    ]

    result_with_doc_id = _prepare_documents_for_chromadb(documents_with_doc_id)
    result_without_doc_id = _prepare_documents_for_chromadb(documents_without_doc_id)

    assert result_with_doc_id.ids == ["custom-id-1", "custom-id-2"]

    assert len(result_without_doc_id.ids) == 2
    # Unique documents get 64-character hashes
    for doc_id in result_without_doc_id.ids:
        assert len(doc_id) == 64, "ID should be 64 characters"
        assert all(c in "0123456789abcdef" for c in doc_id), "ID should be hex"
