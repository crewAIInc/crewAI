"""String templates for A2A (Agent-to-Agent) protocol messaging and status."""

from string import Template
from typing import Final


AVAILABLE_AGENTS_TEMPLATE: Final[Template] = Template(
    "\n<AVAILABLE_A2A_AGENTS>\n    $available_a2a_agents\n</AVAILABLE_A2A_AGENTS>\n"
)
PREVIOUS_A2A_CONVERSATION_TEMPLATE: Final[Template] = Template(
    "\n<PREVIOUS_A2A_CONVERSATION>\n"
    "    $previous_a2a_conversation"
    "\n</PREVIOUS_A2A_CONVERSATION>\n"
)
CONVERSATION_TURN_INFO_TEMPLATE: Final[Template] = Template(
    "\n<CONVERSATION_PROGRESS>\n"
    '    turn="$turn_count"\n'
    '    max_turns="$max_turns"\n'
    "    $warning"
    "\n</CONVERSATION_PROGRESS>\n"
)
UNAVAILABLE_AGENTS_NOTICE_TEMPLATE: Final[Template] = Template(
    "\n<A2A_AGENTS_STATUS>\n"
    "   NOTE: A2A agents were configured but are currently unavailable.\n"
    "   You cannot delegate to remote agents for this task.\n\n"
    "   Unavailable Agents:\n"
    "     $unavailable_agents"
    "\n</A2A_AGENTS_STATUS>\n"
)
REMOTE_AGENT_COMPLETED_NOTICE: Final[str] = """
<REMOTE_AGENT_STATUS>
STATUS: COMPLETED
The remote agent has finished processing your request. Their response is in the conversation history above.
You MUST now:
1. Extract the answer from the conversation history
2. Set is_a2a=false
3. Return the answer as your final message
DO NOT send another request - the task is already done.
</REMOTE_AGENT_STATUS>
"""

REMOTE_AGENT_RESPONSE_NOTICE: Final[str] = """
<REMOTE_AGENT_STATUS>
STATUS: RESPONSE_RECEIVED
The remote agent has responded. Their response is in the conversation history above.

You MUST now:
1. Set is_a2a=false (the remote task is complete and cannot receive more messages)
2. Provide YOUR OWN response to the original task based on the information received

IMPORTANT: Your response should be addressed to the USER who gave you the original task.
Report what the remote agent told you in THIRD PERSON (e.g., "The remote agent said..." or "I learned that...").
Do NOT address the remote agent directly or use "you" to refer to them.
</REMOTE_AGENT_STATUS>
"""
