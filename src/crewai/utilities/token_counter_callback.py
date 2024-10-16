from litellm.integrations.custom_logger import CustomLogger

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess


class TokenCalcHandler(CustomLogger):
    def __init__(self, token_cost_process: TokenProcess):
        self.token_cost_process = token_cost_process

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        if self.token_cost_process is None:
            return

        self.token_cost_process.sum_successful_requests(1)
        self.token_cost_process.sum_prompt_tokens(response_obj["usage"].prompt_tokens)
        self.token_cost_process.sum_completion_tokens(
            response_obj["usage"].completion_tokens
        )
