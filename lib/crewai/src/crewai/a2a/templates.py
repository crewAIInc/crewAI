from string import Template
from typing import Final


AVAILABLE_AGENTS_TEMPLATE: Final[Template] = Template("<AVAILABLE_A2A_AGENTS>\n$available_a2a_agents\n</AVAILABLE_A2A_AGENTS>")
PREVIOUS_A2A_CONVERSATION_TEMPLATE: Final[Template] = Template("\n<PREVIOUS_A2A_CONVERSATION>\n$previous_a2a_conversation\n</PREVIOUS_A2A_CONVERSATION>\n")
