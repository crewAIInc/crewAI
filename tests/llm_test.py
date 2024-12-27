import pytest

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.llm import LLM
from crewai.utilities.token_counter_callback import TokenCalcHandler


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_callback_replacement():
    llm = LLM(model="gpt-4o-mini")

    calc_handler_1 = TokenCalcHandler(token_cost_process=TokenProcess())
    calc_handler_2 = TokenCalcHandler(token_cost_process=TokenProcess())

    llm.call(
        messages=[{"role": "user", "content": "Hello, world!"}],
        callbacks=[calc_handler_1],
    )
    usage_metrics_1 = calc_handler_1.token_cost_process.get_summary()

    llm.call(
        messages=[{"role": "user", "content": "Hello, world from another agent!"}],
        callbacks=[calc_handler_2],
    )
    usage_metrics_2 = calc_handler_2.token_cost_process.get_summary()

    # The first handler should not have been updated
    assert usage_metrics_1.successful_requests == 1
    assert usage_metrics_2.successful_requests == 1
    assert usage_metrics_1 == calc_handler_1.token_cost_process.get_summary()
