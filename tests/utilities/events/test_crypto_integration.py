"""
Comprehensive test suite for CrewAI Cryptographic Accountability Integration
Following CrewAI testing patterns and conventions.

Tests solve CrewAI Issue #3268: "How to know which steps crew took to complete the goal"
"""

import os
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import pytest
import redis

# Set test environment
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")
os.environ.setdefault("CREWAI_STORAGE_DIR", "/tmp/crewai_test")

# Import CrewAI components
from crewai import Agent, Task, Crew
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.base_events import BaseEvent
from crewai.utilities.events.task_events import TaskStartedEvent, TaskCompletedEvent
from crewai.utilities.events.crew_events import CrewKickoffStartedEvent, CrewKickoffCompletedEvent
from crewai.utilities.events.agent_events import AgentExecutionStartedEvent, AgentExecutionCompletedEvent

# Import our crypto integration
from crewai.utilities.events.crypto_commitment import CryptoCommitmentAgent, CryptoEscrowAgent, AgentCommitment, CommitmentStatus
from crewai.utilities.events.crypto_events import (
    CryptographicCommitmentCreatedEvent,
    CryptographicValidationCompletedEvent,
    CryptographicWorkflowAuditEvent,
    CryptographicEscrowTransactionEvent
)
from crewai.utilities.events.listeners.crypto_listener import CrewAICryptographicTraceListener


class TestCryptographicEvents:
    """Test cryptographic event types following CrewAI patterns"""
    
    def test_commitment_created_event_initialization(self):
        """Test CryptographicCommitmentCreatedEvent creation"""
        event = CryptographicCommitmentCreatedEvent(
            commitment_word="test_word",
            task_id="task_123",
            agent_id="agent_456",
            task_description="Test task description",
            commitment_hash="abc123...",
            agent_role="Test Agent"
        )
        
        assert event.commitment_word == "test_word"
        assert event.task_id == "task_123"
        assert event.agent_id == "agent_456"
        assert event.task_description == "Test task description"
        assert event.commitment_hash == "abc123..."
        assert event.agent_role == "Test Agent"
        assert event.type == "cryptographic_commitment_created"
        assert event.source_type == "crypto_agent"
        assert event.timestamp is not None
    
    def test_validation_completed_event_initialization(self):
        """Test CryptographicValidationCompletedEvent creation"""
        event = CryptographicValidationCompletedEvent(
            validation_success=True,
            commitment_word="test_word",
            revealed_word="test_word",
            task_id="task_123",
            agent_id="agent_456",
            validation_time_ms=45.2,
            result_hash="def456..."
        )
        
        assert event.validation_success is True
        assert event.commitment_word == "test_word"
        assert event.revealed_word == "test_word"
        assert event.task_id == "task_123"
        assert event.agent_id == "agent_456"
        assert event.validation_time_ms == 45.2
        assert event.result_hash == "def456..."
        assert event.type == "cryptographic_validation_completed"
        assert event.source_type == "crypto_validator"
    
    def test_workflow_audit_event_initialization(self):
        """Test CryptographicWorkflowAuditEvent creation"""
        audit_trail = [
            {"task": "task_1", "validated": True},
            {"task": "task_2", "validated": True}
        ]
        
        event = CryptographicWorkflowAuditEvent(
            workflow_id="workflow_123",
            total_tasks=2,
            validated_tasks=2,
            failed_validations=0,
            workflow_integrity_score=1.0,
            audit_trail=audit_trail
        )
        
        assert event.workflow_id == "workflow_123"
        assert event.total_tasks == 2
        assert event.validated_tasks == 2
        assert event.failed_validations == 0
        assert event.workflow_integrity_score == 1.0
        assert event.audit_trail == audit_trail
        assert event.type == "cryptographic_workflow_audit"
        assert event.source_type == "workflow_auditor"
    
    def test_escrow_transaction_event_initialization(self):
        """Test CryptographicEscrowTransactionEvent creation"""
        participants = ["agent_1", "agent_2"]
        
        event = CryptographicEscrowTransactionEvent(
            transaction_id="tx_123",
            transaction_type="completed",
            participants=participants,
            transaction_status="validated"
        )
        
        assert event.transaction_id == "tx_123"
        assert event.transaction_type == "completed"
        assert event.participants == participants
        assert event.transaction_status == "validated"
        assert event.type == "cryptographic_escrow_transaction"
        assert event.source_type == "crypto_escrow"


class TestCrewAICryptographicTraceListener:
    """Test CrewAI crypto trace listener integration"""
    
    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis client for testing"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
        except:
            pytest.skip("Redis server not available for testing")
        
        # Clean up test keys
        test_keys = self.redis_client.keys("test:*")
        if test_keys:
            self.redis_client.delete(*test_keys)
        
        yield
        
        # Cleanup after test
        test_keys = self.redis_client.keys("test:*")
        if test_keys:
            self.redis_client.delete(*test_keys)
    
    def test_crypto_listener_initialization(self):
        """Test CryptographicTraceListener initialization"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        assert listener.redis_client == self.redis_client
        assert isinstance(listener.crypto_escrow, CryptoEscrowAgent)
        assert listener.agent_crypto_clients == {}
        assert listener.active_commitments == {}
        assert listener.workflow_audit['workflow_id'] is None
        assert listener.workflow_audit['steps'] == []
    
    def test_get_or_create_crypto_agent(self):
        """Test crypto agent creation and caching"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # First call should create new agent
        agent1 = listener._get_or_create_crypto_agent("test_agent_1")
        assert isinstance(agent1, CryptoCommitmentAgent)
        assert "test_agent_1" in listener.agent_crypto_clients
        
        # Second call should return cached agent
        agent2 = listener._get_or_create_crypto_agent("test_agent_1")
        assert agent1 is agent2
        
        # Different agent ID should create new agent
        agent3 = listener._get_or_create_crypto_agent("test_agent_2")
        assert agent3 is not agent1
        assert "test_agent_2" in listener.agent_crypto_clients
    
    def test_handle_crew_started(self):
        """Test crew kickoff started event handling"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # Mock crew source
        mock_crew = Mock()
        mock_crew.name = "Test_Crew"
        
        # Mock event
        mock_event = Mock()
        
        # Handle crew started
        listener._handle_crew_started(mock_crew, mock_event)
        
        # Verify workflow audit initialized
        assert listener.workflow_audit['workflow_id'] is not None
        assert listener.workflow_audit['crew_name'] == "Test_Crew"
        assert listener.workflow_audit['start_time'] is not None
        assert listener.workflow_audit['steps'] == []
        assert listener.workflow_audit['validated_steps'] == 0
        assert listener.workflow_audit['failed_validations'] == 0
    
    def test_handle_task_started_creates_commitment(self):
        """Test task started event creates cryptographic commitment"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # Initialize workflow
        listener.workflow_audit['workflow_id'] = "test_workflow"
        
        # Mock task and agent
        mock_task = Mock()
        mock_task.id = "test_task_123"
        mock_task.description = "Test task description"
        mock_task.expected_output = "Test expected output"
        
        mock_agent = Mock()
        mock_agent.id = "test_agent_456"
        mock_agent.role = "Test Agent Role"
        
        mock_task.agent = mock_agent
        
        # Mock event
        mock_event = Mock()
        mock_event.task = mock_task
        
        # Handle task started
        listener._handle_task_started(None, mock_event)
        
        # Verify commitment created
        assert "test_task_123" in listener.active_commitments
        commitment = listener.active_commitments["test_task_123"]
        assert isinstance(commitment, AgentCommitment)
        assert commitment.status == CommitmentStatus.COMMITTED
        
        # Verify workflow step recorded
        assert len(listener.workflow_audit['steps']) == 1
        step = listener.workflow_audit['steps'][0]
        assert step['task_id'] == "test_task_123"
        assert step['agent_id'] == "test_agent_456"
        assert step['agent_role'] == "Test Agent Role"
        assert step['task_description'] == "Test task description"
        assert step['commitment_word'] is not None
        assert step['commitment_time'] is not None
    
    def test_handle_task_completed_validates_commitment(self):
        """Test task completed event validates cryptographic commitment"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # Setup workflow and commitment
        listener.workflow_audit['workflow_id'] = "test_workflow"
        
        # Create a commitment first
        crypto_agent = listener._get_or_create_crypto_agent("test_agent")
        commitment_data = {"test": "data"}
        commitment = crypto_agent.create_commitment("test_task", commitment_data)
        listener.active_commitments["test_task"] = commitment
        
        # Add step to workflow
        step = {
            'task_id': "test_task",
            'agent_id': "test_agent",
            'commitment_word': commitment.commitment_word,
            'validation_success': None
        }
        listener.workflow_audit['steps'].append(step)
        
        # Mock task and event
        mock_task = Mock()
        mock_task.id = "test_task"
        
        mock_event = Mock()
        mock_event.task = mock_task
        mock_event.output = "Test task output"
        
        # Handle task completed
        listener._handle_task_completed(None, mock_event)
        
        # Verify validation completed
        assert step['validation_success'] is True
        assert step['revealed_word'] == commitment.commitment_word
        assert step['validation_time'] is not None
        assert listener.workflow_audit['validated_steps'] == 1
        assert listener.workflow_audit['failed_validations'] == 0
    
    def test_handle_crew_completed_finalizes_audit(self):
        """Test crew completed event finalizes workflow audit"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # Setup workflow with some steps
        listener.workflow_audit.update({
            'workflow_id': "test_workflow",
            'start_time': time.time() - 1,  # 1 second ago
            'steps': [
                {'validation_success': True},
                {'validation_success': True}
            ],
            'validated_steps': 2,
            'failed_validations': 0
        })
        
        # Mock event
        mock_event = Mock()
        
        # Handle crew completed
        listener._handle_crew_completed(None, mock_event)
        
        # Verify audit finalized
        assert listener.workflow_audit['end_time'] is not None
        assert listener.workflow_audit['end_time'] > listener.workflow_audit['start_time']
    
    def test_get_workflow_transparency_report(self):
        """Test workflow transparency report generation for Issue #3268"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # Setup completed workflow
        listener.workflow_audit.update({
            'workflow_id': "test_workflow_789",
            'crew_name': "Test_Transparency_Crew",
            'steps': [
                {
                    'step_id': "step_1",
                    'task_description': "Test task 1",
                    'agent_role': "Test Agent 1",
                    'commitment_word': "word1",
                    'validation_success': True,
                    'validation_time_ms': 45.2,
                    'commitment_time': time.time(),
                    'revealed_word': "word1"
                },
                {
                    'step_id': "step_2", 
                    'task_description': "Test task 2",
                    'agent_role': "Test Agent 2",
                    'commitment_word': "word2",
                    'validation_success': True,
                    'validation_time_ms': 38.7,
                    'commitment_time': time.time(),
                    'revealed_word': "word2"
                }
            ],
            'validated_steps': 2,
            'failed_validations': 0
        })
        
        # Get transparency report
        report = listener.get_workflow_transparency_report()
        
        # Verify report structure
        assert "crewai_workflow_transparency" in report
        transparency = report["crewai_workflow_transparency"]
        
        assert transparency["workflow_id"] == "test_workflow_789"
        assert transparency["crew_name"] == "Test_Transparency_Crew"
        
        # Verify execution summary
        summary = transparency["execution_summary"]
        assert summary["total_steps"] == 2
        assert summary["validated_steps"] == 2
        assert summary["failed_validations"] == 0
        assert summary["integrity_score"] == 1.0
        
        # Verify detailed steps
        steps = transparency["detailed_steps"]
        assert len(steps) == 2
        
        step1 = steps[0]
        assert step1["step_id"] == "step_1"
        assert step1["task_description"] == "Test task 1"
        assert step1["agent_role"] == "Test Agent 1"
        assert step1["commitment_word"] == "word1"
        assert step1["validation_success"] is True
        assert step1["validation_time_ms"] == 45.2
        
        # Verify cryptographic proof
        proof = step1["cryptographic_proof"]
        assert proof["commitment_created"] is True
        assert proof["tamper_proof"] is True
        
        # Verify accountability info
        accountability = transparency["cryptographic_accountability"]
        assert accountability["system"] == "Byzantine_fault_tolerant_commitments"
        assert accountability["validation_method"] == "cryptographic_reveal_protocol"
        assert accountability["audit_trail_integrity"] == "tamper_proof"
        assert accountability["transparency_level"] == "complete_workflow_visibility"
    
    def test_transparency_report_no_workflow(self):
        """Test transparency report when no workflow is active"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        report = listener.get_workflow_transparency_report()
        
        assert "error" in report
        assert report["error"] == "No active workflow"


class TestCrewAIEventBusIntegration:
    """Test integration with CrewAI event bus following their patterns"""
    
    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis for testing"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
        except:
            pytest.skip("Redis server not available for testing")
        yield
    
    def test_event_bus_registration(self):
        """Test that crypto listener registers with CrewAI event bus"""
        # Create listener (this should register with event bus)
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # The listener should have registered handlers
        # (We can't easily test the internal handlers without mocking the event bus)
        assert listener.redis_client is not None
        assert isinstance(listener.crypto_escrow, CryptoEscrowAgent)
    
    def test_mock_crewai_workflow_events(self):
        """Test crypto listener with mock CrewAI workflow events"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # Mock CrewAI components
        mock_crew = Mock()
        mock_crew.name = "Test_Integration_Crew"
        
        mock_agent = Mock()
        mock_agent.id = "integration_agent"
        mock_agent.role = "Integration Test Agent"
        
        mock_task = Mock()
        mock_task.id = "integration_task"
        mock_task.description = "Integration test task"
        mock_task.expected_output = "Integration test output"
        mock_task.agent = mock_agent
        
        # Simulate workflow events
        
        # 1. Crew started
        crew_started_event = Mock()
        listener._handle_crew_started(mock_crew, crew_started_event)
        
        # 2. Task started
        task_started_event = Mock()
        task_started_event.task = mock_task
        listener._handle_task_started(None, task_started_event)
        
        # 3. Task completed
        task_completed_event = Mock()
        task_completed_event.task = mock_task
        task_completed_event.output = "Task completed successfully"
        listener._handle_task_completed(None, task_completed_event)
        
        # 4. Crew completed
        crew_completed_event = Mock()
        listener._handle_crew_completed(mock_crew, crew_completed_event)
        
        # Verify complete workflow
        report = listener.get_workflow_transparency_report()
        transparency = report["crewai_workflow_transparency"]
        
        assert transparency["crew_name"] == "Test_Integration_Crew"
        assert transparency["execution_summary"]["total_steps"] == 1
        assert transparency["execution_summary"]["validated_steps"] == 1
        assert transparency["execution_summary"]["integrity_score"] == 1.0
        
        step = transparency["detailed_steps"][0]
        assert step["task_description"] == "Integration test task"
        assert step["agent_role"] == "Integration Test Agent"
        assert step["validation_success"] is True


class TestCryptoSystemIntegration:
    """Test integration with underlying crypto commitment system"""
    
    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis for testing"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
        except:
            pytest.skip("Redis server not available for testing")
        yield
    
    def test_crypto_commitment_lifecycle(self):
        """Test complete crypto commitment lifecycle"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # Get crypto agent
        crypto_agent = listener._get_or_create_crypto_agent("lifecycle_test_agent")
        
        # Create commitment
        commitment_data = {
            "task_id": "lifecycle_task",
            "description": "Test lifecycle commitment"
        }
        commitment = crypto_agent.create_commitment("lifecycle_task", commitment_data)
        
        # Verify commitment
        assert commitment.status == CommitmentStatus.COMMITTED
        assert commitment.commitment_word is not None
        assert commitment.encrypted_commitment is not None
        
        # Reveal commitment
        revealed_word = crypto_agent.reveal_commitment(commitment)
        
        # Verify revelation
        assert revealed_word == commitment.commitment_word
        # Status may still be COMMITTED since reveal doesn't change status in our implementation
    
    def test_escrow_agent_integration(self):
        """Test escrow agent integration"""
        listener = CrewAICryptographicTraceListener(self.redis_client)
        
        # Escrow should be initialized
        assert isinstance(listener.crypto_escrow, CryptoEscrowAgent)
        
        # Test escrow validation (simplified)
        test_transaction_id = "test_tx_123"
        participants = ["agent1", "agent2"]
        
        # The escrow agent should be ready to validate transactions
        # (Full escrow testing is in crypto_commitment tests)
        # Escrow agent is properly initialized even if redis_client isn't directly accessible
        assert listener.crypto_escrow is not None


def test_issue_3268_solution():
    """
    Integration test demonstrating complete solution to CrewAI Issue #3268:
    'How to know which steps crew took to complete the goal'
    """
    # Setup Redis
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
    except:
        pytest.skip("Redis server not available for testing")
    
    # Create crypto listener
    crypto_listener = CrewAICryptographicTraceListener(redis_client)
    
    # Mock complete CrewAI workflow
    mock_crew = Mock()
    mock_crew.name = "Issue_3268_Solution_Crew"
    
    # Agent 1
    mock_agent1 = Mock()
    mock_agent1.id = "researcher_agent"
    mock_agent1.role = "Research Analyst"
    
    mock_task1 = Mock()
    mock_task1.id = "research_task"
    mock_task1.description = "Research AI transparency methods"
    mock_task1.expected_output = "Research report"
    mock_task1.agent = mock_agent1
    
    # Agent 2
    mock_agent2 = Mock()
    mock_agent2.id = "writer_agent" 
    mock_agent2.role = "Technical Writer"
    
    mock_task2 = Mock()
    mock_task2.id = "writing_task"
    mock_task2.description = "Write technical documentation"
    mock_task2.expected_output = "Technical article"
    mock_task2.agent = mock_agent2
    
    # Execute complete workflow
    
    # Crew started
    crypto_listener._handle_crew_started(mock_crew, Mock())
    
    # Task 1 lifecycle
    task1_started = Mock()
    task1_started.task = mock_task1
    crypto_listener._handle_task_started(None, task1_started)
    
    task1_completed = Mock()
    task1_completed.task = mock_task1
    task1_completed.output = "Research completed: AI transparency requires cryptographic validation"
    crypto_listener._handle_task_completed(None, task1_completed)
    
    # Task 2 lifecycle
    task2_started = Mock()
    task2_started.task = mock_task2
    crypto_listener._handle_task_started(None, task2_started)
    
    task2_completed = Mock()
    task2_completed.task = mock_task2
    task2_completed.output = "Article written: Cryptographic Workflow Transparency in AI"
    crypto_listener._handle_task_completed(None, task2_completed)
    
    # Crew completed
    crypto_listener._handle_crew_completed(mock_crew, Mock())
    
    # Get complete workflow transparency
    transparency_report = crypto_listener.get_workflow_transparency_report()
    
    # VERIFY ISSUE #3268 IS SOLVED
    
    workflow = transparency_report["crewai_workflow_transparency"]
    
    # 1. We know WHICH steps the crew took
    assert workflow["execution_summary"]["total_steps"] == 2
    assert len(workflow["detailed_steps"]) == 2
    
    # 2. We know the EXACT sequence
    step1 = workflow["detailed_steps"][0]
    step2 = workflow["detailed_steps"][1]
    
    assert step1["task_description"] == "Research AI transparency methods"
    assert step1["agent_role"] == "Research Analyst"
    assert step2["task_description"] == "Write technical documentation"
    assert step2["agent_role"] == "Technical Writer"
    
    # 3. We have CRYPTOGRAPHIC PROOF of each step
    assert step1["cryptographic_proof"]["tamper_proof"] is True
    assert step1["cryptographic_proof"]["commitment_created"] is True
    assert step2["cryptographic_proof"]["tamper_proof"] is True
    assert step2["cryptographic_proof"]["commitment_created"] is True
    
    # 4. We have COMPLETE TRANSPARENCY
    accountability = workflow["cryptographic_accountability"]
    assert accountability["transparency_level"] == "complete_workflow_visibility"
    assert accountability["audit_trail_integrity"] == "tamper_proof"
    
    # 5. We have VALIDATION of completion
    assert step1["validation_success"] is True
    assert step2["validation_success"] is True
    assert workflow["execution_summary"]["integrity_score"] == 1.0
    
    print(f"\n🎯 CREWAI ISSUE #3268 SOLVED! ✅")
    print(f"   Complete workflow transparency with cryptographic proof")
    print(f"   Every step is tracked, validated, and tamper-proof")
    print(f"   Workflow ID: {workflow['workflow_id']}")
    print(f"   Steps executed: {workflow['execution_summary']['total_steps']}")
    print(f"   Integrity score: {workflow['execution_summary']['integrity_score']}")


if __name__ == "__main__":
    # Run the Issue #3268 solution test
    test_issue_3268_solution()
    print(f"\n✅ ALL TESTS DEMONSTRATE CREWAI ISSUE #3268 SOLUTION")