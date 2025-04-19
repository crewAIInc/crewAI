from unittest.mock import MagicMock, patch

import pytest
from mem0.memory.main import Memory

from crewai.memory.user.user_memory import UserMemory
from crewai.memory.user.user_memory_item import UserMemoryItem


class MockCrew:
    def __init__(self, memory_config):
        self.memory_config = memory_config

@pytest.fixture
def user_memory():
    """Fixture to create a UserMemory instance"""
    crew = MockCrew(
        memory_config={
            "provider": "mem0",
            "config": {"user_id": "john"},
            "user_memory" : {}
        }
    )

    user_memory = MagicMock(spec=UserMemory)

    with patch.object(Memory,'__new__',return_value=user_memory):
        user_memory_instance = UserMemory(crew=crew)
    
    return user_memory_instance

def test_save_and_search(user_memory):
    memory = UserMemoryItem(
        data="""test value test value test value test value test value test value
        test value test value test value test value test value test value
        test value test value test value test value test value test value""",
        user="test_user",
        metadata={"task": "test_task"},
    )

    with patch.object(UserMemory, "save") as mock_save:
        user_memory.save(
            value=memory.data,
            metadata=memory.metadata,
            user=memory.user
        )

        mock_save.assert_called_once_with(
            value=memory.data,
            metadata=memory.metadata,
            user=memory.user
        )

    expected_result = [
        {
            "context": memory.data,
            "metadata": {"agent": "test_agent"},
            "score": 0.95,
        }
    ]
    expected_result = ["mocked_result"]

    # Use patch.object to mock UserMemory's search method
    with patch.object(UserMemory, 'search', return_value=expected_result) as mock_search:
        find = UserMemory.search("test value", score_threshold=0.01)[0]
        mock_search.assert_called_once_with("test value", score_threshold=0.01)
        assert find == expected_result[0]
