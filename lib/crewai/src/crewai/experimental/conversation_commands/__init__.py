"""Experimental ``/btw`` commands for conversational Flows.

Conversational turns treat every user line as a new graph run. This add-on
lets you interject a side instruction that steers routing and replies
without appending a user message. It is opt-in and not part of the stable
``handle_turn`` / ``chat`` contract.

Example::

    from crewai.experimental.conversation_commands import btw_commands
    from crewai.flow import ConversationConfig, ConversationState, Flow, listen


    @btw_commands
    @ConversationConfig()
    class SupportFlow(Flow[ConversationState]):
        @listen("RESEARCH")
        def handle_research(self) -> str:
            return "researching"


    flow = SupportFlow()
    flow.handle_turn("/btw keep answers under 20 words")
    flow.handle_turn("/btw route RESEARCH")
    flow.handle_turn("latest AI news")
"""

from crewai.experimental.conversation_commands.addon import (
    btw_commands,
    enable_btw_commands,
)
from crewai.experimental.conversation_commands.parser import (
    HELP_TEXT,
    BtwAction,
    BtwKind,
    ParsedBtwLine,
    parse_btw_line,
)
from crewai.experimental.conversation_commands.steering import (
    BtwSteering,
    get_btw_steering,
)


__all__ = [
    "HELP_TEXT",
    "BtwAction",
    "BtwKind",
    "BtwSteering",
    "ParsedBtwLine",
    "btw_commands",
    "enable_btw_commands",
    "get_btw_steering",
    "parse_btw_line",
]
