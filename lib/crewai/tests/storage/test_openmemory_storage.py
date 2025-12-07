import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class MockOpenMemory:
    def __init__(self, mode=None, path=None, tier=None, embeddings=None):
        self.mode = mode
        self.path = path
        self.tier = tier
        self.embeddings = embeddings
        self._memories: list[dict] = []

    def add(self, content, userId=None, tags=None, metadata=None, **kwargs):
        memory_id = f"mem_{len(self._memories)}"
        self._memories.append(
            {
                "id": memory_id,
                "content": content,
                "userId": userId,
                "tags": tags,
                "metadata": metadata,
            }
        )
        return {"id": memory_id, "primarySector": "semantic", "sectors": ["semantic"]}

    def query(self, query, k=10, filters=None, **kwargs):
        results = []
        for mem in self._memories:
            if query.lower() in mem["content"].lower():
                results.append(
                    {
                        "content": mem["content"],
                        "score": 0.9,
                        "metadata": mem["metadata"],
                    }
                )
        return results[:k]

    def close(self):
        pass


class MockCrew:
    def __init__(self):
        self.agents = [MagicMock(role="Test Agent")]


@pytest.fixture
def mock_openmemory():
    return MockOpenMemory


@pytest.fixture
def temp_storage_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_memory.sqlite")


@pytest.fixture
def openmemory_storage_with_mock(mock_openmemory, temp_storage_path):
    with patch(
        "crewai.memory.storage.openmemory_storage.OpenMemory", mock_openmemory
    ):
        from crewai.memory.storage.openmemory_storage import OpenMemoryStorage

        config = {
            "path": temp_storage_path,
            "tier": "fast",
            "embeddings": {"provider": "synthetic"},
            "user_id": "test_user",
        }
        crew = MockCrew()
        storage = OpenMemoryStorage(type="external", crew=crew, config=config)
        return storage


def test_openmemory_storage_initialization(openmemory_storage_with_mock):
    storage = openmemory_storage_with_mock
    assert storage.memory_type == "external"
    assert storage.tier == "fast"
    assert storage.user_id == "test_user"
    assert storage.embeddings == {"provider": "synthetic"}


def test_openmemory_storage_invalid_type():
    with patch(
        "crewai.memory.storage.openmemory_storage.OpenMemory", MockOpenMemory
    ):
        from crewai.memory.storage.openmemory_storage import OpenMemoryStorage

        with pytest.raises(ValueError, match="Invalid type"):
            OpenMemoryStorage(type="invalid_type", config={"path": "/tmp/test.sqlite"})


def test_openmemory_storage_save(openmemory_storage_with_mock):
    storage = openmemory_storage_with_mock

    test_value = "This is a test memory about AI agents"
    test_metadata = {"description": "Test description", "agent": "Test Agent"}

    storage.save(test_value, test_metadata)

    assert len(storage.memory._memories) == 1
    saved_memory = storage.memory._memories[0]
    assert saved_memory["content"] == test_value
    assert saved_memory["userId"] == "test_user"
    assert saved_memory["metadata"]["type"] == "external"
    assert saved_memory["metadata"]["description"] == "Test description"


def test_openmemory_storage_save_with_tags(openmemory_storage_with_mock):
    storage = openmemory_storage_with_mock

    test_value = "Memory with tags"
    test_metadata = {"tags": ["tag1", "tag2"], "description": "Tagged memory"}

    storage.save(test_value, test_metadata)

    saved_memory = storage.memory._memories[0]
    assert saved_memory["tags"] == ["tag1", "tag2"]


def test_openmemory_storage_search(openmemory_storage_with_mock):
    storage = openmemory_storage_with_mock

    storage.save("Memory about Python programming", {"description": "Python"})
    storage.save("Memory about JavaScript", {"description": "JavaScript"})
    storage.save("Memory about Python frameworks", {"description": "Frameworks"})

    results = storage.search("Python", limit=5, score_threshold=0.5)

    assert len(results) == 2
    assert all("content" in r for r in results)
    assert any("Python programming" in r["content"] for r in results)
    assert any("Python frameworks" in r["content"] for r in results)


def test_openmemory_storage_search_with_score_threshold(openmemory_storage_with_mock):
    storage = openmemory_storage_with_mock

    storage.save("Test memory", {"description": "Test"})

    results = storage.search("Test", limit=5, score_threshold=0.95)
    assert len(results) == 0

    results = storage.search("Test", limit=5, score_threshold=0.5)
    assert len(results) == 1


def test_openmemory_storage_reset(temp_storage_path):
    with patch(
        "crewai.memory.storage.openmemory_storage.OpenMemory", MockOpenMemory
    ):
        from crewai.memory.storage.openmemory_storage import OpenMemoryStorage

        config = {
            "path": temp_storage_path,
            "tier": "fast",
            "embeddings": {"provider": "synthetic"},
        }
        storage = OpenMemoryStorage(type="external", config=config)

        storage.save("Test memory", {"description": "Test"})
        assert len(storage.memory._memories) == 1

        storage.reset()

        assert len(storage.memory._memories) == 0


def test_openmemory_storage_default_path():
    with patch(
        "crewai.memory.storage.openmemory_storage.OpenMemory", MockOpenMemory
    ):
        from crewai.memory.storage.openmemory_storage import OpenMemoryStorage

        config = {
            "tier": "fast",
            "embeddings": {"provider": "synthetic"},
        }
        storage = OpenMemoryStorage(type="external", config=config)

        assert storage.path is not None
        assert "openmemory_external.sqlite" in storage.path


def test_openmemory_storage_different_memory_types():
    with patch(
        "crewai.memory.storage.openmemory_storage.OpenMemory", MockOpenMemory
    ):
        from crewai.memory.storage.openmemory_storage import OpenMemoryStorage

        for memory_type in ["short_term", "long_term", "entities", "external"]:
            config = {
                "tier": "fast",
                "embeddings": {"provider": "synthetic"},
            }
            storage = OpenMemoryStorage(type=memory_type, config=config)
            assert storage.memory_type == memory_type
            assert f"openmemory_{memory_type}.sqlite" in storage.path


def test_openmemory_import_error():
    with patch.dict("sys.modules", {"openmemory": None}):
        with patch(
            "crewai.memory.storage.openmemory_storage.OpenMemory",
            side_effect=ImportError("No module named 'openmemory'"),
        ):
            pass


def test_external_memory_openmemory_provider():
    with patch(
        "crewai.memory.storage.openmemory_storage.OpenMemory", MockOpenMemory
    ):
        from crewai.memory.external.external_memory import ExternalMemory

        supported = ExternalMemory.external_supported_storages()
        assert "openmemory" in supported


def test_external_memory_create_openmemory_storage(temp_storage_path):
    with patch(
        "crewai.memory.storage.openmemory_storage.OpenMemory", MockOpenMemory
    ):
        from crewai.memory.external.external_memory import ExternalMemory

        crew = MockCrew()
        embedder_config = {
            "provider": "openmemory",
            "config": {
                "path": temp_storage_path,
                "tier": "fast",
                "embeddings": {"provider": "synthetic"},
            },
        }

        storage = ExternalMemory.create_storage(crew, embedder_config)

        assert storage is not None
        assert storage.memory_type == "external"
