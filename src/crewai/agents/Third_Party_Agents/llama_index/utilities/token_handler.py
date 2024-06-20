from typing import Any, Dict, Optional
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.callbacks.schema import CBEventType
from crewai.agents.third_party_agents.utilities.token_process import TokenProcess


class ExtendedTokenCountingHandler(TokenCountingHandler):
    def __init__(self, tokenizer, token_process: TokenProcess):
        super().__init__(tokenizer)
        self.token_process = token_process

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().on_event_end(event_type, payload, event_id, **kwargs)

        if event_type == CBEventType.LLM and payload is not None:
            last_event = self.llm_token_counts[-1]
            self.token_process.sum_prompt_tokens(last_event.prompt_token_count)
            self.token_process.sum_completion_tokens(last_event.completion_token_count)
            self.token_process.sum_successful_requests(1)
