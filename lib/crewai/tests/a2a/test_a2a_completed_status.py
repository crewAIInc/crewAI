"""Test A2A delegation properly handles 'completed' status without looping."""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from crewai import Agent, Task
from crewai.a2a.config import A2AConfig

try:
    from a2a.types import AgentCard, Message, Part, Role, TextPart

    A2A_SDK_INSTALLED = True
except ImportError:
    A2A_SDK_INSTALLED = False


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_a2a_delegation_stops_on_completed_status():
    """Test that A2A delegation stops immediately when remote agent returns 'completed' status.
    
    This test verifies the fix for issue #3899 where the server agent was ignoring
    the 'completed' status and delegating the same request again, causing an infinite loop.
    """
    a2a_config = A2AConfig(
        endpoint="http://test-endpoint.com",
        max_turns=10,
    )
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        a2a=a2a_config,
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    final_message_text = "This is the final answer from the remote agent"
    mock_history = [
        Message(
            role=Role.user,
            message_id=str(uuid4()),
            parts=[Part(root=TextPart(text="Initial request"))],
        ),
        Message(
            role=Role.agent,
            message_id=str(uuid4()),
            parts=[Part(root=TextPart(text=final_message_text))],
        ),
    ]
    
    mock_a2a_result = {
        "status": "completed",
        "result": final_message_text,
        "history": mock_history,
        "agent_card": MagicMock(spec=AgentCard),
    }
    
    mock_agent_card = MagicMock(spec=AgentCard)
    mock_agent_card.name = "Test Remote Agent"
    mock_agent_card.url = "http://test-endpoint.com"
    
    with patch("crewai.a2a.wrapper.execute_a2a_delegation") as mock_execute:
        with patch("crewai.a2a.wrapper.fetch_agent_card", return_value=mock_agent_card):
            with patch("crewai.a2a.wrapper._handle_agent_response_and_continue") as mock_handle:
                mock_execute.return_value = mock_a2a_result
                
                from crewai.a2a.wrapper import _delegate_to_a2a
                
                mock_agent_response = Mock()
                mock_agent_response.is_a2a = True
                mock_agent_response.a2a_ids = ["http://test-endpoint.com/"]
                mock_agent_response.message = "Please delegate this task"
                
                result = _delegate_to_a2a(
                    self=agent,
                    agent_response=mock_agent_response,
                    task=task,
                    original_fn=Mock(),
                    context=None,
                    tools=None,
                    agent_cards={"http://test-endpoint.com/": mock_agent_card},
                    original_task_description="Test task",
                )
                
                assert mock_execute.call_count == 1, (
                    f"execute_a2a_delegation should be called exactly once, "
                    f"but was called {mock_execute.call_count} times"
                )
                
                assert mock_handle.call_count == 0, (
                    "_handle_agent_response_and_continue should NOT be called "
                    "when status is 'completed'"
                )
                
                assert result == final_message_text


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_a2a_delegation_continues_on_input_required():
    """Test that A2A delegation continues when remote agent returns 'input_required' status.
    
    This test verifies that the 'input_required' status still triggers the LLM
    to decide on next steps, unlike 'completed' which should return immediately.
    """
    a2a_config = A2AConfig(
        endpoint="http://test-endpoint.com",
        max_turns=10,
    )
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        a2a=a2a_config,
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    mock_history_1 = [
        Message(
            role=Role.user,
            message_id=str(uuid4()),
            parts=[Part(root=TextPart(text="Initial request"))],
        ),
        Message(
            role=Role.agent,
            message_id=str(uuid4()),
            parts=[Part(root=TextPart(text="I need more information"))],
        ),
    ]
    
    mock_history_2 = [
        *mock_history_1,
        Message(
            role=Role.user,
            message_id=str(uuid4()),
            parts=[Part(root=TextPart(text="Here is the additional info"))],
        ),
        Message(
            role=Role.agent,
            message_id=str(uuid4()),
            parts=[Part(root=TextPart(text="Final answer with all info"))],
        ),
    ]
    
    mock_a2a_result_1 = {
        "status": "input_required",
        "error": "I need more information",
        "history": mock_history_1,
        "agent_card": MagicMock(spec=AgentCard),
    }
    
    mock_a2a_result_2 = {
        "status": "completed",
        "result": "Final answer with all info",
        "history": mock_history_2,
        "agent_card": MagicMock(spec=AgentCard),
    }
    
    mock_agent_card = MagicMock(spec=AgentCard)
    mock_agent_card.name = "Test Remote Agent"
    mock_agent_card.url = "http://test-endpoint.com"
    
    with patch("crewai.a2a.wrapper.execute_a2a_delegation") as mock_execute:
        with patch("crewai.a2a.wrapper.fetch_agent_card", return_value=mock_agent_card):
            with patch("crewai.a2a.wrapper._handle_agent_response_and_continue") as mock_handle:
                mock_execute.side_effect = [mock_a2a_result_1, mock_a2a_result_2]
                
                mock_handle.return_value = (None, "Here is the additional info")
                
                from crewai.a2a.wrapper import _delegate_to_a2a
                
                mock_agent_response = Mock()
                mock_agent_response.is_a2a = True
                mock_agent_response.a2a_ids = ["http://test-endpoint.com/"]
                mock_agent_response.message = "Please delegate this task"
                
                result = _delegate_to_a2a(
                    self=agent,
                    agent_response=mock_agent_response,
                    task=task,
                    original_fn=Mock(),
                    context=None,
                    tools=None,
                    agent_cards={"http://test-endpoint.com/": mock_agent_card},
                    original_task_description="Test task",
                )
                
                assert mock_execute.call_count == 2, (
                    f"execute_a2a_delegation should be called twice, "
                    f"but was called {mock_execute.call_count} times"
                )
                
                assert mock_handle.call_count == 1, (
                    "_handle_agent_response_and_continue should be called once "
                    "for 'input_required' status"
                )
                
                assert result == "Final answer with all info"


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_a2a_delegation_completed_with_empty_history():
    """Test that A2A delegation handles 'completed' status with empty history gracefully.
    
    This test verifies that when the remote agent returns 'completed' but the history
    is empty or doesn't contain an agent message, we still return a reasonable result.
    """
    a2a_config = A2AConfig(
        endpoint="http://test-endpoint.com",
        max_turns=10,
    )
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        a2a=a2a_config,
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    mock_a2a_result = {
        "status": "completed",
        "result": "",  # Empty result
        "history": [],  # Empty history
        "agent_card": MagicMock(spec=AgentCard),
    }
    
    mock_agent_card = MagicMock(spec=AgentCard)
    mock_agent_card.name = "Test Remote Agent"
    mock_agent_card.url = "http://test-endpoint.com"
    
    with patch("crewai.a2a.wrapper.execute_a2a_delegation") as mock_execute:
        with patch("crewai.a2a.wrapper.fetch_agent_card", return_value=mock_agent_card):
            with patch("crewai.a2a.wrapper._handle_agent_response_and_continue") as mock_handle:
                mock_execute.return_value = mock_a2a_result
                
                from crewai.a2a.wrapper import _delegate_to_a2a
                
                mock_agent_response = Mock()
                mock_agent_response.is_a2a = True
                mock_agent_response.a2a_ids = ["http://test-endpoint.com/"]
                mock_agent_response.message = "Please delegate this task"
                
                result = _delegate_to_a2a(
                    self=agent,
                    agent_response=mock_agent_response,
                    task=task,
                    original_fn=Mock(),
                    context=None,
                    tools=None,
                    agent_cards={"http://test-endpoint.com/": mock_agent_card},
                    original_task_description="Test task",
                )
                
                assert mock_execute.call_count == 1
                
                assert mock_handle.call_count == 0
                
                assert result == "Conversation completed"


@pytest.mark.skipif(not A2A_SDK_INSTALLED, reason="Requires a2a-sdk to be installed")
def test_a2a_delegation_completed_extracts_from_history():
    """Test that A2A delegation extracts final message from history when result is empty.
    
    This test verifies that when the remote agent returns 'completed' with an empty result
    but has messages in the history, we extract the final agent message from history.
    """
    a2a_config = A2AConfig(
        endpoint="http://test-endpoint.com",
        max_turns=10,
    )
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        a2a=a2a_config,
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    
    final_message_text = "Final message from history"
    mock_history = [
        Message(
            role=Role.user,
            message_id=str(uuid4()),
            parts=[Part(root=TextPart(text="Initial request"))],
        ),
        Message(
            role=Role.agent,
            message_id=str(uuid4()),
            parts=[Part(root=TextPart(text=final_message_text))],
        ),
    ]
    
    mock_a2a_result = {
        "status": "completed",
        "result": "",  # Empty result, should extract from history
        "history": mock_history,
        "agent_card": MagicMock(spec=AgentCard),
    }
    
    mock_agent_card = MagicMock(spec=AgentCard)
    mock_agent_card.name = "Test Remote Agent"
    mock_agent_card.url = "http://test-endpoint.com"
    
    with patch("crewai.a2a.wrapper.execute_a2a_delegation") as mock_execute:
        with patch("crewai.a2a.wrapper.fetch_agent_card", return_value=mock_agent_card):
            with patch("crewai.a2a.wrapper._handle_agent_response_and_continue") as mock_handle:
                mock_execute.return_value = mock_a2a_result
                
                from crewai.a2a.wrapper import _delegate_to_a2a
                
                mock_agent_response = Mock()
                mock_agent_response.is_a2a = True
                mock_agent_response.a2a_ids = ["http://test-endpoint.com/"]
                mock_agent_response.message = "Please delegate this task"
                
                result = _delegate_to_a2a(
                    self=agent,
                    agent_response=mock_agent_response,
                    task=task,
                    original_fn=Mock(),
                    context=None,
                    tools=None,
                    agent_cards={"http://test-endpoint.com/": mock_agent_card},
                    original_task_description="Test task",
                )
                
                assert mock_execute.call_count == 1
                
                assert mock_handle.call_count == 0
                
                assert result == final_message_text
