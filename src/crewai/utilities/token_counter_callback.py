import warnings

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Usage

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess


class TokenCalcHandler(CustomLogger):
    def __init__(self, token_cost_process: TokenProcess):
        self.token_cost_process = token_cost_process

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        if self.token_cost_process is None:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            usage: Usage = response_obj["usage"]
            self.token_cost_process.sum_successful_requests(1)
            self.token_cost_process.sum_prompt_tokens(usage.prompt_tokens)
            self.token_cost_process.sum_completion_tokens(usage.completion_tokens)
            if usage.prompt_tokens_details:
                self.token_cost_process.sum_cached_prompt_tokens(
                    usage.prompt_tokens_details.cached_tokens
                )
