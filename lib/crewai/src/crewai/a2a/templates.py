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
