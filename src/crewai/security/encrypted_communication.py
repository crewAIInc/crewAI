"""
Encrypted Communication Module

This module provides functionality for encrypting and decrypting agent-to-agent
communication in CrewAI. It leverages existing security infrastructure including
agent fingerprints and Fernet encryption.
"""

import json
import logging
from typing import Any, Dict, Optional, Union

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field

from crewai.security.fingerprint import Fingerprint

logger = logging.getLogger(__name__)


class EncryptedMessage(BaseModel):
    """
    Represents an encrypted message between agents.
    
    Attributes:
        encrypted_payload (str): The encrypted message content
        sender_fingerprint (str): The fingerprint of the sending agent
        recipient_fingerprint (str): The fingerprint of the intended recipient
        message_type (str): The type of message (task, question, response, etc.)
    """
    encrypted_payload: str = Field(..., description="The encrypted message content")
    sender_fingerprint: str = Field(..., description="Sender agent's fingerprint")
    recipient_fingerprint: str = Field(..., description="Recipient agent's fingerprint") 
    message_type: str = Field(default="communication", description="Type of message")


class AgentCommunicationEncryption:
    """
    Handles encryption and decryption of agent-to-agent communication.
    
    Uses Fernet symmetric encryption with keys derived from agent fingerprints.
    Provides methods to encrypt and decrypt communication payloads.
    """
    
    def __init__(self, agent_fingerprint: Fingerprint):
        """
        Initialize encryption handler for an agent.
        
        Args:
            agent_fingerprint (Fingerprint): The agent's unique fingerprint
        """
        self.agent_fingerprint = agent_fingerprint
        self._encryption_keys: Dict[str, Fernet] = {}
        
    def _derive_communication_key(self, sender_fp: str, recipient_fp: str) -> bytes:
        """
        Derive a communication key from sender and recipient fingerprints.
        
        Creates a deterministic key based on both agent fingerprints to ensure
        both agents can derive the same key for encrypted communication.
        
        Args:
            sender_fp (str): Sender agent's fingerprint
            recipient_fp (str): Recipient agent's fingerprint
            
        Returns:
            bytes: 32-byte encryption key for Fernet
        """
        # Sort fingerprints to ensure consistent key derivation regardless of role
        fp_pair = tuple(sorted([sender_fp, recipient_fp]))
        key_material = f"crewai_comm_{fp_pair[0]}_{fp_pair[1]}".encode('utf-8')
        
        # Use SHA-256 to derive a 32-byte key from the fingerprint pair
        import hashlib
        key_hash = hashlib.sha256(key_material).digest()
        
        # Fernet requires base64-encoded 32-byte key
        import base64
        return base64.urlsafe_b64encode(key_hash)
        
    def _get_fernet(self, sender_fp: str, recipient_fp: str) -> Fernet:
        """
        Get or create Fernet instance for communication between two agents.
        
        Args:
            sender_fp (str): Sender agent's fingerprint
            recipient_fp (str): Recipient agent's fingerprint
            
        Returns:
            Fernet: Encryption instance for this agent pair
        """
        # Create cache key from sorted fingerprints
        cache_key = "_".join(sorted([sender_fp, recipient_fp]))
        
        if cache_key not in self._encryption_keys:
            key = self._derive_communication_key(sender_fp, recipient_fp)
            self._encryption_keys[cache_key] = Fernet(key)
            
        return self._encryption_keys[cache_key]
    
    def encrypt_message(
        self, 
        message: Union[str, Dict[str, Any]], 
        recipient_fingerprint: Fingerprint,
        message_type: str = "communication"
    ) -> EncryptedMessage:
        """
        Encrypt a message for a specific recipient agent.
        
        Args:
            message (Union[str, Dict[str, Any]]): The message to encrypt
            recipient_fingerprint (Fingerprint): The recipient agent's fingerprint
            message_type (str): Type of message being sent
            
        Returns:
            EncryptedMessage: Encrypted message container
            
        Raises:
            ValueError: If encryption fails
        """
        try:
            # Convert message to JSON string if it's a dict
            if isinstance(message, dict):
                message_str = json.dumps(message)
            else:
                message_str = str(message)
                
            # Get Fernet instance for this communication pair
            fernet = self._get_fernet(
                self.agent_fingerprint.uuid_str, 
                recipient_fingerprint.uuid_str
            )
            
            # Encrypt the message
            encrypted_bytes = fernet.encrypt(message_str.encode('utf-8'))
            encrypted_payload = encrypted_bytes.decode('utf-8')
            
            logger.debug(f"Encrypted message from {self.agent_fingerprint.uuid_str[:8]}... to {recipient_fingerprint.uuid_str[:8]}...")
            
            return EncryptedMessage(
                encrypted_payload=encrypted_payload,
                sender_fingerprint=self.agent_fingerprint.uuid_str,
                recipient_fingerprint=recipient_fingerprint.uuid_str,
                message_type=message_type
            )
            
        except Exception as e:
            logger.error(f"Failed to encrypt message: {e}")
            raise ValueError(f"Message encryption failed: {e}")
    
    def decrypt_message(self, encrypted_message: EncryptedMessage) -> Union[str, Dict[str, Any]]:
        """
        Decrypt a message intended for this agent.
        
        Args:
            encrypted_message (EncryptedMessage): The encrypted message to decrypt
            
        Returns:
            Union[str, Dict[str, Any]]: The decrypted message content
            
        Raises:
            ValueError: If decryption fails or message is not for this agent
        """
        try:
            # Verify this message is intended for this agent
            if encrypted_message.recipient_fingerprint != self.agent_fingerprint.uuid_str:
                raise ValueError(f"Message not intended for this agent. Expected {self.agent_fingerprint.uuid_str[:8]}..., got {encrypted_message.recipient_fingerprint[:8]}...")
            
            # Get Fernet instance for this communication pair
            fernet = self._get_fernet(
                encrypted_message.sender_fingerprint,
                encrypted_message.recipient_fingerprint
            )
            
            # Decrypt the message
            decrypted_bytes = fernet.decrypt(encrypted_message.encrypted_payload.encode('utf-8'))
            decrypted_str = decrypted_bytes.decode('utf-8')
            
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str
                
        except Exception as e:
            logger.error(f"Failed to decrypt message: {e}")
            raise ValueError(f"Message decryption failed: {e}")
    
    def is_encrypted_communication(self, message: Any) -> bool:
        """
        Check if a message is an encrypted communication.
        
        Args:
            message (Any): Message to check
            
        Returns:
            bool: True if message is encrypted, False otherwise
        """
        return isinstance(message, (EncryptedMessage, dict)) and (
            hasattr(message, 'encrypted_payload') or 
            (isinstance(message, dict) and 'encrypted_payload' in message)
        )