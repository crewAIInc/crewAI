# Encrypted Agent Communication

CrewAI now supports encrypted agent-to-agent communication to ensure secure information exchange between agents in multi-agent workflows.

## Features

- **Fernet Encryption**: Uses industry-standard Fernet symmetric encryption
- **Fingerprint-based Key Derivation**: Unique keys derived from agent fingerprints  
- **Secure Message Routing**: Messages can only be decrypted by intended recipients
- **Backward Compatible**: Non-encrypted agents continue to work normally
- **Optional**: Encryption can be enabled per-agent or per-crew

## Quick Start

### 1. Enable Encryption for Agents

```python
from crewai import Agent
from crewai.security import SecurityConfig

# Create agents with encryption enabled
researcher = Agent(
    role="Research Analyst",
    goal="Conduct research and analysis",
    backstory="Expert researcher with deep analytical skills",
    security_config=SecurityConfig(encrypted_communication=True)
)

writer = Agent(
    role="Content Writer", 
    goal="Create engaging content",
    backstory="Skilled writer who creates compelling narratives",
    security_config=SecurityConfig(encrypted_communication=True)
)
```

### 2. Use Encrypted Communication in Tasks

```python
from crewai import Crew, Task
from crewai.tools.agent_tools import AskQuestionTool

# Create tools that support encrypted communication
agent_tools = [
    AskQuestionTool(
        agents=[researcher, writer],
        description="Ask questions to team members with encrypted communication"
    )
]

# Create tasks that will use encrypted communication
research_task = Task(
    description="Research the latest AI trends",
    expected_output="Comprehensive research report",
    agent=researcher
)

writing_task = Task(
    description="Ask the researcher about their findings and create a blog post",
    expected_output="Engaging blog post based on research", 
    agent=writer,
    tools=agent_tools  # These will use encrypted communication automatically
)

# Run the crew - communications will be automatically encrypted
crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff()
```

## How It Works

### Security Architecture

1. **Agent Fingerprints**: Each agent gets a unique cryptographic fingerprint
2. **Key Derivation**: Communication keys are derived from sender/recipient fingerprint pairs
3. **Message Encryption**: Payloads are encrypted using Fernet with derived keys
4. **Secure Routing**: Only intended recipients can decrypt messages

### Message Flow

```
Sender Agent → Encrypt Message → Encrypted Payload → Recipient Agent → Decrypt Message
     ↓              ↓                    ↓                  ↓              ↓
  Fingerprint   Derive Key       EncryptedMessage    Derive Key    Original Message
```

### Encryption Details

- **Algorithm**: Fernet (AES 128 in CBC mode with HMAC-SHA256)
- **Key Length**: 256-bit encryption keys
- **Key Derivation**: SHA-256 based on sorted agent fingerprints
- **Message Format**: JSON with encrypted payload and metadata

## Configuration Options

### SecurityConfig Parameters

```python
from crewai.security import SecurityConfig

# Enable encryption
security_config = SecurityConfig(encrypted_communication=True)

# Disable encryption (default)
security_config = SecurityConfig(encrypted_communication=False)
```

### Mixed Encryption Scenarios

```python
# Some agents with encryption, others without
encrypted_agent = Agent(
    role="Secure Agent",
    security_config=SecurityConfig(encrypted_communication=True),
    # ... other params
)

plain_agent = Agent(
    role="Regular Agent", 
    security_config=SecurityConfig(encrypted_communication=False),
    # ... other params
)

# Agent tools automatically handle mixed scenarios:
# - Encrypted agents → Encrypted communication
# - Non-encrypted agents → Plain communication  
# - Mixed → Falls back to plain communication with warning
```

## Security Considerations

### What Is Protected

✅ **Task descriptions and context** passed between agents  
✅ **Questions and responses** in agent-to-agent communication  
✅ **Delegation payloads** including sensitive instructions  
✅ **Agent-to-agent metadata** like sender/recipient information  

### What Is NOT Protected

❌ **Agent configurations** (roles, goals, backstories)  
❌ **Task outputs** stored in crew results  
❌ **LLM API calls** to external services  
❌ **Tool executions** outside of agent communication  

### Best Practices

1. **Enable encryption** for sensitive workflows
2. **Use unique fingerprints** per deployment/environment
3. **Monitor logs** for encryption failures or downgrades
4. **Test mixed scenarios** with both encrypted and non-encrypted agents
5. **Keep fingerprints secure** - they are used for key derivation

## Advanced Usage

### Direct Encryption API

For advanced use cases, you can use the encryption API directly:

```python
from crewai.security import AgentCommunicationEncryption, Fingerprint

# Create encryption handlers
sender_fp = Fingerprint()
recipient_fp = Fingerprint()
sender_encryption = AgentCommunicationEncryption(sender_fp)
recipient_encryption = AgentCommunicationEncryption(recipient_fp)

# Encrypt a message
message = {"task": "Analyze data", "context": "Q4 results"}
encrypted_msg = sender_encryption.encrypt_message(
    message, 
    recipient_fp, 
    message_type="analysis_request"
)

# Decrypt the message
decrypted_msg = recipient_encryption.decrypt_message(encrypted_msg)
```

### Custom Fingerprints

```python
# Generate deterministic fingerprints from seeds
fp = Fingerprint.generate(seed="agent-role-environment")

# Use custom metadata
fp = Fingerprint.generate(
    seed="unique-seed",
    metadata={"environment": "production", "version": "1.0"}
)
```

## Troubleshooting

### Common Issues

**Q: "Message not intended for this agent" error**  
A: This happens when an agent tries to decrypt a message meant for another agent. Check that the correct recipient agent is being used.

**Q: "Encryption failed, falling back to plain communication" warning**  
A: This indicates the encryption process failed and the system fell back to unencrypted communication. Check agent security configurations.

**Q: Mixed encrypted/non-encrypted agents not working**  
A: Ensure at least one agent has encryption enabled for the encryption to activate. If no agents have encryption, all communication will be plain text.

### Debug Logging

Enable debug logging to see encryption activities:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Look for log messages like:
# DEBUG:crewai.security.encrypted_communication:Encrypted message from abc12345... to def67890...
# DEBUG:crewai.tools.agent_tools.base_agent_tools:Executing encrypted communication task...
```

## Performance Impact

- **Encryption/Decryption**: Minimal overhead (~1-2ms per message)
- **Key Derivation**: Cached after first use per agent pair
- **Memory**: Small increase for encryption handlers and cached keys
- **Network**: No additional network calls (all local encryption)

## Migration Guide

### From Non-Encrypted to Encrypted

1. Add `security_config` to your agent definitions
2. No code changes required for agent tools
3. Test with mixed encrypted/non-encrypted agents first
4. Enable encryption for all agents in production

```python
# Before
agent = Agent(role="Analyst", goal="...", backstory="...")

# After  
agent = Agent(
    role="Analyst", 
    goal="...", 
    backstory="...",
    security_config=SecurityConfig(encrypted_communication=True)
)
```

That's it! Your agents will automatically use encrypted communication when both sender and recipient support it.