import logging
from typing import Optional, Union, Dict, Any

from pydantic import Field

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities import I18N
from crewai.security import AgentCommunicationEncryption, EncryptedMessage

logger = logging.getLogger(__name__)


class BaseAgentTool(BaseTool):
    """Base class for agent-related tools with optional encrypted communication support"""

    agents: list[BaseAgent] = Field(description="List of available agents")
    i18n: I18N = Field(
        default_factory=I18N, description="Internationalization settings"
    )

    def __init__(self, **data):
        """Initialize BaseAgentTool with optional encryption support."""
        super().__init__(**data)
        self._encryption_handler: Optional[AgentCommunicationEncryption] = None
    
    @property
    def encryption_enabled(self) -> bool:
        """Check if encryption is enabled for agent communication."""
        # Check if any agent has encryption enabled
        return any(
            hasattr(agent, 'security_config') and 
            agent.security_config and
            getattr(agent.security_config, 'encrypted_communication', False)
            for agent in self.agents
        )
    
    def _get_encryption_handler(self, sender_agent: BaseAgent) -> Optional[AgentCommunicationEncryption]:
        """Get encryption handler for a specific agent."""
        if not hasattr(sender_agent, 'security_config') or not sender_agent.security_config:
            return None
            
        if not getattr(sender_agent.security_config, 'encrypted_communication', False):
            return None
            
        # Create encryption handler if it doesn't exist
        if self._encryption_handler is None:
            self._encryption_handler = AgentCommunicationEncryption(
                sender_agent.security_config.fingerprint
            )
        
        return self._encryption_handler
    
    def _prepare_communication_payload(
        self, 
        sender_agent: BaseAgent,
        recipient_agent: BaseAgent, 
        task: str, 
        context: Optional[str] = None
    ) -> Union[Dict[str, Any], EncryptedMessage]:
        """
        Prepare communication payload, with optional encryption.
        
        Args:
            sender_agent: The agent sending the communication
            recipient_agent: The agent receiving the communication
            task: The task or question to communicate
            context: Optional context for the communication
            
        Returns:
            Union[Dict[str, Any], EncryptedMessage]: Plain or encrypted message
        """
        # Prepare the base message
        message_payload = {
            "task": task,
            "context": context or "",
            "sender_role": getattr(sender_agent, 'role', 'unknown'),
            "message_type": "agent_communication"
        }
        
        # Check if encryption should be used
        encryption_handler = self._get_encryption_handler(sender_agent)
        if encryption_handler and hasattr(recipient_agent, 'security_config') and recipient_agent.security_config:
            try:
                logger.info(f"Starting encrypted communication from '{sender_agent.role}' to '{recipient_agent.role}'")
                # Encrypt the message for the recipient
                encrypted_msg = encryption_handler.encrypt_message(
                    message_payload,
                    recipient_agent.security_config.fingerprint,
                    message_type="agent_communication"
                )
                logger.info(f"Encrypted communication established between '{sender_agent.role}' and '{recipient_agent.role}'")
                logger.debug(f"Encrypted communication from {sender_agent.role} to {recipient_agent.role}")
                return encrypted_msg
            except Exception as e:
                logger.warning(f"Encryption failed, falling back to plain communication: {e}")
        
        return message_payload
    
    def _process_received_communication(
        self, 
        recipient_agent: BaseAgent, 
        message: Union[str, Dict[str, Any], EncryptedMessage]
    ) -> Union[str, Dict[str, Any]]:
        """
        Process received communication, with optional decryption.
        
        Args:
            recipient_agent: The agent receiving the communication
            message: The message to process (may be encrypted)
            
        Returns:
            Union[str, Dict[str, Any]]: Processed message content
        """
        # Handle encrypted messages
        if isinstance(message, EncryptedMessage) or (
            isinstance(message, dict) and 'encrypted_payload' in message
        ):
            encryption_handler = self._get_encryption_handler(recipient_agent)
            if encryption_handler:
                try:
                    logger.info(f"Starting decryption of received communication for '{recipient_agent.role}'")
                    # Convert dict to EncryptedMessage if needed
                    if isinstance(message, dict):
                        message = EncryptedMessage(**message)
                    
                    decrypted = encryption_handler.decrypt_message(message)
                    logger.info(f"Successfully decrypted communication for '{recipient_agent.role}'")
                    logger.debug(f"Decrypted communication for {recipient_agent.role}")
                    return decrypted
                except Exception as e:
                    logger.error(f"Decryption failed for {recipient_agent.role}: {e}")
                    raise ValueError(f"Failed to decrypt communication: {e}")
            else:
                logger.warning(f"Received encrypted message but {recipient_agent.role} has no decryption capability")
                raise ValueError("Received encrypted message but agent cannot decrypt it")
        
        # Return message as-is for plain communication
        return message

    def sanitize_agent_name(self, name: str) -> str:
        """
        Sanitize agent role name by normalizing whitespace and setting to lowercase.
        Converts all whitespace (including newlines) to single spaces and removes quotes.

        Args:
            name (str): The agent role name to sanitize

        Returns:
            str: The sanitized agent role name, with whitespace normalized,
                 converted to lowercase, and quotes removed
        """
        if not name:
            return ""
        # Normalize all whitespace (including newlines) to single spaces
        normalized = " ".join(name.split())
        # Remove quotes and convert to lowercase
        return normalized.replace('"', "").casefold()

    def _get_coworker(self, coworker: Optional[str], **kwargs) -> Optional[str]:
        coworker = coworker or kwargs.get("co_worker") or kwargs.get("coworker")
        if coworker:
            is_list = coworker.startswith("[") and coworker.endswith("]")
            if is_list:
                coworker = coworker[1:-1].split(",")[0]
        return coworker

    def _execute(
        self,
        agent_name: Optional[str],
        task: str,
        context: Optional[str] = None
    ) -> str:
        """
        Execute delegation to an agent with case-insensitive and whitespace-tolerant matching.
        Supports both encrypted and non-encrypted communication based on agent configuration.

        Args:
            agent_name: Name/role of the agent to delegate to (case-insensitive)
            task: The specific question or task to delegate
            context: Optional additional context for the task execution

        Returns:
            str: The execution result from the delegated agent or an error message
                 if the agent cannot be found
        """
        try:
            if agent_name is None:
                agent_name = ""
                logger.debug("No agent name provided, using empty string")

            # It is important to remove the quotes from the agent name.
            # The reason we have to do this is because less-powerful LLM's
            # have difficulty producing valid JSON.
            # As a result, we end up with invalid JSON that is truncated like this:
            # {"task": "....", "coworker": "....
            # when it should look like this:
            # {"task": "....", "coworker": "...."}
            sanitized_name = self.sanitize_agent_name(agent_name)
            logger.debug(f"Sanitized agent name from '{agent_name}' to '{sanitized_name}'")

            available_agents = [agent.role for agent in self.agents]
            logger.debug(f"Available agents: {available_agents}")

            agent = [  # type: ignore # Incompatible types in assignment (expression has type "list[BaseAgent]", variable has type "str | None")
                available_agent
                for available_agent in self.agents
                if self.sanitize_agent_name(available_agent.role) == sanitized_name
            ]
            logger.debug(f"Found {len(agent)} matching agents for role '{sanitized_name}'")
        except (AttributeError, ValueError) as e:
            # Handle specific exceptions that might occur during role name processing
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join(
                    [f"- {self.sanitize_agent_name(agent.role)}" for agent in self.agents]
                ),
                error=str(e)
            )

        if not agent:
            # No matching agent found after sanitization
            return self.i18n.errors("agent_tool_unexisting_coworker").format(
                coworkers="\n".join(
                    [f"- {self.sanitize_agent_name(agent.role)}" for agent in self.agents]
                ),
                error=f"No agent found with role '{sanitized_name}'"
            )

        target_agent = agent[0]
        
        # Determine sender agent (first agent with security config, or first agent as fallback)
        sender_agent = None
        for a in self.agents:
            if hasattr(a, 'security_config') and a.security_config:
                sender_agent = a
                break
        if not sender_agent:
            sender_agent = self.agents[0] if self.agents else target_agent

        try:
            # Prepare communication with optional encryption
            communication_payload = self._prepare_communication_payload(
                sender_agent=sender_agent,
                recipient_agent=target_agent,
                task=task,
                context=context
            )
            
            # Create task for execution
            task_with_assigned_agent = Task(
                description=task,
                agent=target_agent,
                expected_output=target_agent.i18n.slice("manager_request"),
                i18n=target_agent.i18n,
            )
            
            # Execute with processed communication context
            if isinstance(communication_payload, EncryptedMessage):
                logger.info(f"Executing encrypted communication task for agent '{self.sanitize_agent_name(target_agent.role)}'")
                logger.debug(f"Executing encrypted communication task for agent '{self.sanitize_agent_name(target_agent.role)}'")
                # For encrypted messages, pass the encrypted payload as additional context
                # The target agent will need to handle decryption during execution
                enhanced_context = f"ENCRYPTED_COMMUNICATION: {communication_payload.model_dump_json()}"
                if context:
                    enhanced_context += f"\nADDITIONAL_CONTEXT: {context}"
                result = target_agent.execute_task(task_with_assigned_agent, enhanced_context)
            else:
                logger.info(f"Executing plain communication task for agent '{self.sanitize_agent_name(target_agent.role)}'")
                logger.debug(f"Executing plain communication task for agent '{self.sanitize_agent_name(target_agent.role)}'")
                result = target_agent.execute_task(task_with_assigned_agent, context)
            
            return result
            
        except Exception as e:
            # Handle task creation or execution errors
            logger.error(f"Task execution failed for agent '{self.sanitize_agent_name(target_agent.role)}': {e}")
            return self.i18n.errors("agent_tool_execution_error").format(
                agent_role=self.sanitize_agent_name(target_agent.role),
                error=str(e)
            )
