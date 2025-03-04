import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks.base import BaseCallbackHandler
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Usage

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess


class AbstractTokenCounter(ABC):
    """
    Abstract base class for token counting callbacks.
    Implementations should track token usage from different LLM providers.
    """

    def __init__(self, token_process: Optional[TokenProcess] = None):
        """Initialize with a TokenProcess instance to track tokens."""
        self.token_process = token_process

    @abstractmethod
    def update_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token usage counts in the token process."""
        pass


class LiteLLMTokenCounter(CustomLogger, AbstractTokenCounter):
    """
    Token counter implementation for LiteLLM.
    Uses LiteLLM's CustomLogger interface to track token usage.
    """

    def __init__(self, token_process: Optional[TokenProcess] = None):
        AbstractTokenCounter.__init__(self, token_process)
        CustomLogger.__init__(self)

    def update_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token usage counts in the token process."""
        if self.token_process is None:
            return

        if prompt_tokens > 0:
            self.token_process.sum_prompt_tokens(prompt_tokens)

        if completion_tokens > 0:
            self.token_process.sum_completion_tokens(completion_tokens)

        self.token_process.sum_successful_requests(1)

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Process successful LLM call and extract token usage information.
        This method is called by LiteLLM after a successful completion.
        """
        if self.token_process is None:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if isinstance(response_obj, dict) and "usage" in response_obj:
                usage: Usage = response_obj["usage"]
                if usage:
                    prompt_tokens = 0
                    completion_tokens = 0

                    if hasattr(usage, "prompt_tokens"):
                        prompt_tokens = usage.prompt_tokens
                    elif isinstance(usage, dict) and "prompt_tokens" in usage:
                        prompt_tokens = usage["prompt_tokens"]

                    if hasattr(usage, "completion_tokens"):
                        completion_tokens = usage.completion_tokens
                    elif isinstance(usage, dict) and "completion_tokens" in usage:
                        completion_tokens = usage["completion_tokens"]

                    self.update_token_usage(prompt_tokens, completion_tokens)

                    # Handle cached tokens if available
                    if (
                        hasattr(usage, "prompt_tokens_details")
                        and usage.prompt_tokens_details
                        and usage.prompt_tokens_details.cached_tokens
                    ):
                        self.token_process.sum_cached_prompt_tokens(
                            usage.prompt_tokens_details.cached_tokens
                        )


class LangChainTokenCounter(BaseCallbackHandler, AbstractTokenCounter):
    """
    Token counter implementation for LangChain.
    Implements the necessary callback methods to track token usage from LangChain responses.
    """

    def __init__(self, token_process: Optional[TokenProcess] = None):
        BaseCallbackHandler.__init__(self)
        AbstractTokenCounter.__init__(self, token_process)

    def update_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token usage counts in the token process."""
        if self.token_process is None:
            return

        if prompt_tokens > 0:
            self.token_process.sum_prompt_tokens(prompt_tokens)

        if completion_tokens > 0:
            self.token_process.sum_completion_tokens(completion_tokens)

        self.token_process.sum_successful_requests(1)

    @property
    def ignore_llm(self) -> bool:
        return False

    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_agent(self) -> bool:
        return False

    @property
    def ignore_chat_model(self) -> bool:
        return False

    @property
    def ignore_retriever(self) -> bool:
        return True

    @property
    def ignore_tools(self) -> bool:
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts processing."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when LLM generates a new token."""
        pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """
        Called when LLM ends processing.
        Extracts token usage from LangChain response objects.
        """
        if self.token_process is None:
            return

        # Handle LangChain response format
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage", {})

            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            self.update_token_usage(prompt_tokens, completion_tokens)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when LLM errors."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Called when a chain starts."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when a chain ends."""
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when a chain errors."""
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Called when a tool starts."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool ends."""
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when a tool errors."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Called when text is generated."""
        pass

    def on_agent_start(self, serialized: Dict[str, Any], **kwargs: Any) -> None:
        """Called when an agent starts."""
        pass

    def on_agent_end(self, output: Any, **kwargs: Any) -> None:
        """Called when an agent ends."""
        pass

    def on_agent_error(self, error: BaseException, **kwargs: Any) -> None:
        """Called when an agent errors."""
        pass


# For backward compatibility
class TokenCalcHandler(LiteLLMTokenCounter):
    """
    Alias for LiteLLMTokenCounter.
    """

    pass
