"""String templates for A2A (Agent-to-Agent) delegation prompts."""

from string import Template
from typing import Final


AVAILABLE_AGENTS_TEMPLATE: Final[Template] = Template(
    "\n<AVAILABLE_A2A_AGENTS>\n"
    "You can delegate to remote agents using the delegate_to_* tools below. "
    "Each tool's description lists the remote agent's capabilities — call the "
    "tool whose capabilities best match the task. Pass the question or sub-task "
    "to the remote agent via the tool's `message` argument; the tool returns "
    "the remote agent's response, which you should incorporate into your final "
    "answer. If the available agents are not a good fit, answer directly "
    "without calling a delegation tool.\n\n"
    "    $available_a2a_agents"
    "\n</AVAILABLE_A2A_AGENTS>\n"
)
UNAVAILABLE_AGENTS_NOTICE_TEMPLATE: Final[Template] = Template(
    "\n<A2A_AGENTS_STATUS>\n"
    "   NOTE: A2A agents were configured but are currently unavailable.\n"
    "   You cannot delegate to remote agents for this task.\n\n"
    "   Unavailable Agents:\n"
    "     $unavailable_agents"
    "\n</A2A_AGENTS_STATUS>\n"
)
