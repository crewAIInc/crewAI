"""
Encryption events for agent-to-agent communication
"""

from typing import Optional
from crewai.agents.agent_builder.base_agent import BaseAgent
from .base_events import BaseEvent


class EncryptionStartedEvent(BaseEvent):
    """Event emitted when agent-to-agent encryption starts"""

    sender_agent: BaseAgent
    recipient_agent: BaseAgent
    message_type: str = "agent_communication"
    type: str = "encryption_started"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the sender agent
        if hasattr(self.sender_agent, "fingerprint") and self.sender_agent.fingerprint:
            self.source_fingerprint = self.sender_agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.sender_agent.fingerprint, "metadata")
                and self.sender_agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.sender_agent.fingerprint.metadata


class EncryptionCompletedEvent(BaseEvent):
    """Event emitted when agent-to-agent encryption completes successfully"""

    sender_agent: BaseAgent
    recipient_agent: BaseAgent
    message_type: str = "agent_communication"
    type: str = "encryption_completed"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the sender agent
        if hasattr(self.sender_agent, "fingerprint") and self.sender_agent.fingerprint:
            self.source_fingerprint = self.sender_agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.sender_agent.fingerprint, "metadata")
                and self.sender_agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.sender_agent.fingerprint.metadata


class DecryptionStartedEvent(BaseEvent):
    """Event emitted when agent-to-agent decryption starts"""

    recipient_agent: BaseAgent
    sender_fingerprint: str
    message_type: str = "agent_communication"
    type: str = "decryption_started"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the recipient agent
        if hasattr(self.recipient_agent, "fingerprint") and self.recipient_agent.fingerprint:
            self.source_fingerprint = self.recipient_agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.recipient_agent.fingerprint, "metadata")
                and self.recipient_agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.recipient_agent.fingerprint.metadata


class DecryptionCompletedEvent(BaseEvent):
    """Event emitted when agent-to-agent decryption completes successfully"""

    recipient_agent: BaseAgent
    sender_fingerprint: str
    message_type: str = "agent_communication"
    type: str = "decryption_completed"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the recipient agent
        if hasattr(self.recipient_agent, "fingerprint") and self.recipient_agent.fingerprint:
            self.source_fingerprint = self.recipient_agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.recipient_agent.fingerprint, "metadata")
                and self.recipient_agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.recipient_agent.fingerprint.metadata


class EncryptedCommunicationStartedEvent(BaseEvent):
    """Event emitted when encrypted communication between agents begins"""

    sender_agent: BaseAgent
    recipient_agent: BaseAgent
    type: str = "encrypted_communication_started"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the sender agent
        if hasattr(self.sender_agent, "fingerprint") and self.sender_agent.fingerprint:
            self.source_fingerprint = self.sender_agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.sender_agent.fingerprint, "metadata")
                and self.sender_agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.sender_agent.fingerprint.metadata


class EncryptedCommunicationEstablishedEvent(BaseEvent):
    """Event emitted when encrypted communication is successfully established"""

    sender_agent: BaseAgent
    recipient_agent: BaseAgent
    type: str = "encrypted_communication_established"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the sender agent
        if hasattr(self.sender_agent, "fingerprint") and self.sender_agent.fingerprint:
            self.source_fingerprint = self.sender_agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.sender_agent.fingerprint, "metadata")
                and self.sender_agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.sender_agent.fingerprint.metadata


class EncryptedTaskExecutionEvent(BaseEvent):
    """Event emitted when an encrypted communication task is being executed"""

    agent: BaseAgent
    task_type: str = "encrypted_communication"
    type: str = "encrypted_task_execution"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the agent
        if hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata