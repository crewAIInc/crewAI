from string import Template
from typing import Final


AVAILABLE_AGENTS_TEMPLATE: Final[Template] = Template(
    "<AVAILABLE_A2A_AGENTS>\n$available_a2a_agents\n</AVAILABLE_A2A_AGENTS>"
)
PREVIOUS_A2A_CONVERSATION_TEMPLATE: Final[Template] = Template(
    "\n<PREVIOUS_A2A_CONVERSATION>\n$previous_a2a_conversation\n</PREVIOUS_A2A_CONVERSATION>\n"
)
CONVERSATION_TURN_INFO_TEMPLATE: Final[Template] = Template(
    '\n<CONVERSATION_PROGRESS>\nturn="$turn_count" max_turns="$max_turns">\n$warning</CONVERSATION_PROGRESS>\n'
)
UNAVAILABLE_AGENTS_NOTICE_TEMPLATE: Final[Template] = Template(
    "\n\n<A2A_AGENTS_STATUS>\n\n"
    "   NOTE: A2A agents were configured but are currently unavailable.\n"
    "   You cannot delegate to remote agents for this task.\n\n"
    "   Unavailable Agents:\n"
    "     $unavailable_agents"
    "\n</A2A_AGENTS_STATUS>\n"
)
