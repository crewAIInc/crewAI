"""Test encrypted agent communication functionality."""

import json
import pytest
from unittest.mock import Mock, patch

from crewai.security import (
    AgentCommunicationEncryption, 
    EncryptedMessage, 
    Fingerprint,
    SecurityConfig
)
from crewai.agent import Agent
from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool
from crewai.tools.agent_tools.delegate_work_tool import DelegateWorkTool


class TestAgentCommunicationEncryption:
    """Test the encryption/decryption functionality."""
    
    def test_encryption_initialization(self):
        """Test initialization of encryption handler."""
        fp = Fingerprint()
        encryption = AgentCommunicationEncryption(fp)
        
        assert encryption.agent_fingerprint == fp
        assert encryption._encryption_keys == {}
    
    def test_key_derivation_consistency(self):
        """Test that key derivation is consistent for same agent pair."""
        fp1 = Fingerprint()
        fp2 = Fingerprint()
        
        encryption1 = AgentCommunicationEncryption(fp1)
        encryption2 = AgentCommunicationEncryption(fp2)
        
        # Keys should be the same regardless of which agent derives them
        key1_from_1 = encryption1._derive_communication_key(fp1.uuid_str, fp2.uuid_str)
        key1_from_2 = encryption2._derive_communication_key(fp1.uuid_str, fp2.uuid_str)
        key2_from_1 = encryption1._derive_communication_key(fp2.uuid_str, fp1.uuid_str)
        key2_from_2 = encryption2._derive_communication_key(fp2.uuid_str, fp1.uuid_str)
        
        assert key1_from_1 == key1_from_2
        assert key1_from_1 == key2_from_1
        assert key1_from_1 == key2_from_2
    
    def test_message_encryption_decryption(self):
        """Test basic message encryption and decryption."""
        sender_fp = Fingerprint()
        recipient_fp = Fingerprint()
        
        sender_encryption = AgentCommunicationEncryption(sender_fp)
        recipient_encryption = AgentCommunicationEncryption(recipient_fp)
        
        original_message = "Hello, this is a test message"
        
        # Encrypt message
        encrypted_msg = sender_encryption.encrypt_message(
            original_message, 
            recipient_fp, 
            "test_message"
        )
        
        assert isinstance(encrypted_msg, EncryptedMessage)
        assert encrypted_msg.sender_fingerprint == sender_fp.uuid_str
        assert encrypted_msg.recipient_fingerprint == recipient_fp.uuid_str
        assert encrypted_msg.message_type == "test_message"
        assert encrypted_msg.encrypted_payload != original_message
        
        # Decrypt message
        decrypted_message = recipient_encryption.decrypt_message(encrypted_msg)
        
        assert decrypted_message == original_message
    
    def test_dict_message_encryption_decryption(self):
        """Test encryption and decryption of dictionary messages."""
        sender_fp = Fingerprint()
        recipient_fp = Fingerprint()
        
        sender_encryption = AgentCommunicationEncryption(sender_fp)
        recipient_encryption = AgentCommunicationEncryption(recipient_fp)
        
        original_message = {
            "task": "Analyze this data",
            "context": "This is important context",
            "priority": "high"
        }
        
        # Encrypt message
        encrypted_msg = sender_encryption.encrypt_message(
            original_message, 
            recipient_fp
        )
        
        # Decrypt message
        decrypted_message = recipient_encryption.decrypt_message(encrypted_msg)
        
        assert decrypted_message == original_message
    
    def test_wrong_recipient_decryption_fails(self):
        """Test that decryption fails for wrong recipient."""
        sender_fp = Fingerprint()
        recipient_fp = Fingerprint()
        wrong_recipient_fp = Fingerprint()
        
        sender_encryption = AgentCommunicationEncryption(sender_fp)
        wrong_recipient_encryption = AgentCommunicationEncryption(wrong_recipient_fp)
        
        original_message = "Secret message"
        
        # Encrypt for correct recipient
        encrypted_msg = sender_encryption.encrypt_message(
            original_message, 
            recipient_fp
        )
        
        # Try to decrypt with wrong recipient
        with pytest.raises(ValueError, match="Message not intended for this agent"):
            wrong_recipient_encryption.decrypt_message(encrypted_msg)
    
    def test_is_encrypted_communication(self):
        """Test detection of encrypted communication."""
        fp = Fingerprint()
        encryption = AgentCommunicationEncryption(fp)
        
        # Test with EncryptedMessage
        encrypted_msg = EncryptedMessage(
            encrypted_payload="test",
            sender_fingerprint="sender",
            recipient_fingerprint="recipient"
        )
        assert encryption.is_encrypted_communication(encrypted_msg) is True
        
        # Test with dict containing encrypted_payload
        encrypted_dict = {
            "encrypted_payload": "test",
            "sender_fingerprint": "sender"
        }
        assert encryption.is_encrypted_communication(encrypted_dict) is True
        
        # Test with regular message
        regular_msg = "Plain text message"
        assert encryption.is_encrypted_communication(regular_msg) is False
        
        # Test with regular dict
        regular_dict = {"task": "Do something"}
        assert encryption.is_encrypted_communication(regular_dict) is False


class TestSecurityConfigEncryption:
    """Test SecurityConfig encryption settings."""
    
    def test_security_config_encryption_default(self):
        """Test default encryption setting."""
        config = SecurityConfig()
        assert config.encrypted_communication is False
    
    def test_security_config_encryption_enabled(self):
        """Test enabling encryption."""
        config = SecurityConfig(encrypted_communication=True)
        assert config.encrypted_communication is True


class TestAgentToolsEncryption:
    """Test encrypted communication in agent tools."""
    
    @pytest.fixture
    def agents_with_encryption(self):
        """Create test agents with encryption enabled."""
        # Create agents with security configs
        sender_agent = Mock()
        sender_agent.role = "sender"
        sender_agent.security_config = SecurityConfig(encrypted_communication=True)
        sender_agent.i18n = Mock()
        
        recipient_agent = Mock()
        recipient_agent.role = "recipient" 
        recipient_agent.security_config = SecurityConfig(encrypted_communication=True)
        recipient_agent.i18n = Mock()
        recipient_agent.i18n.slice.return_value = "Expected output"
        recipient_agent.execute_task = Mock(return_value="Task completed")
        
        return [sender_agent, recipient_agent]
    
    @pytest.fixture
    def agents_without_encryption(self):
        """Create test agents without encryption."""
        sender_agent = Mock()
        sender_agent.role = "sender"
        sender_agent.security_config = SecurityConfig(encrypted_communication=False)
        sender_agent.i18n = Mock()
        
        recipient_agent = Mock()
        recipient_agent.role = "recipient"
        recipient_agent.security_config = SecurityConfig(encrypted_communication=False)  
        recipient_agent.i18n = Mock()
        recipient_agent.i18n.slice.return_value = "Expected output"
        recipient_agent.execute_task = Mock(return_value="Task completed")
        
        return [sender_agent, recipient_agent]
    
    def test_encryption_enabled_detection(self, agents_with_encryption):
        """Test detection of encryption capability."""
        tool = AskQuestionTool(agents=agents_with_encryption, description="Test tool")
        assert tool.encryption_enabled is True
    
    def test_encryption_disabled_detection(self, agents_without_encryption):
        """Test detection when encryption is disabled."""
        tool = AskQuestionTool(agents=agents_without_encryption, description="Test tool")
        assert tool.encryption_enabled is False
    
    def test_prepare_encrypted_communication_payload(self, agents_with_encryption):
        """Test preparation of encrypted communication payload."""
        sender, recipient = agents_with_encryption
        tool = AskQuestionTool(agents=[sender, recipient], description="Test tool")
        
        payload = tool._prepare_communication_payload(
            sender_agent=sender,
            recipient_agent=recipient,
            task="Test task",
            context="Test context"
        )
        
        assert isinstance(payload, EncryptedMessage)
        assert payload.sender_fingerprint == sender.security_config.fingerprint.uuid_str
        assert payload.recipient_fingerprint == recipient.security_config.fingerprint.uuid_str
    
    def test_prepare_plain_communication_payload(self, agents_without_encryption):
        """Test preparation of plain communication payload."""
        sender, recipient = agents_without_encryption
        tool = AskQuestionTool(agents=[sender, recipient], description="Test tool")
        
        payload = tool._prepare_communication_payload(
            sender_agent=sender,
            recipient_agent=recipient,
            task="Test task", 
            context="Test context"
        )
        
        assert isinstance(payload, dict)
        assert payload["task"] == "Test task"
        assert payload["context"] == "Test context"
        assert payload["sender_role"] == "sender"
    
    def test_execute_with_encryption_enabled(self, agents_with_encryption):
        """Test task execution with encryption enabled."""
        sender, recipient = agents_with_encryption
        tool = AskQuestionTool(agents=[sender, recipient], description="Test tool")
        
        result = tool._run(
            question="What is AI?",
            context="Test context",
            coworker="recipient"
        )
        
        # Verify task was executed
        assert result == "Task completed"
        
        # Verify execute_task was called with encrypted context
        recipient.execute_task.assert_called_once()
        call_args = recipient.execute_task.call_args
        context_arg = call_args[0][1]  # Second argument is context
        
        assert "ENCRYPTED_COMMUNICATION:" in context_arg
    
    def test_execute_with_encryption_disabled(self, agents_without_encryption):
        """Test task execution with encryption disabled."""
        sender, recipient = agents_without_encryption
        tool = AskQuestionTool(agents=[sender, recipient], description="Test tool")
        
        result = tool._run(
            question="What is AI?",
            context="Test context", 
            coworker="recipient"
        )
        
        # Verify task was executed
        assert result == "Task completed"
        
        # Verify execute_task was called with plain context
        recipient.execute_task.assert_called_once()
        call_args = recipient.execute_task.call_args
        context_arg = call_args[0][1]  # Second argument is context
        
        assert context_arg == "Test context"


class TestBackwardCompatibility:
    """Test that existing functionality still works."""
    
    def test_agents_without_security_config_work(self):
        """Test that agents without security config still function."""
        # Create agents without security config
        agent1 = Mock()
        agent1.role = "agent1"
        agent1.i18n = Mock()
        agent1.i18n.slice.return_value = "Expected output"
        agent1.execute_task = Mock(return_value="Task completed")
        # No security_config attribute
        
        agent2 = Mock()
        agent2.role = "agent2"
        agent2.i18n = Mock()
        
        tool = AskQuestionTool(agents=[agent1, agent2], description="Test tool")
        
        result = tool._run(
            question="Test question",
            context="Test context",
            coworker="agent1"
        )
        
        assert result == "Task completed"
        agent1.execute_task.assert_called_once()