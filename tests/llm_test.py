from time import sleep

import pytest

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.llm import LLM
from crewai.utilities.token_counter_callback import TokenCalcHandler


# TODO: This test fails without print statement, which makes me think that something is happening asynchronously that we need to eventually fix and dive deeper into at a later date
@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_callback_replacement():
    llm1 = LLM(model="gpt-4o-mini")
    llm2 = LLM(model="gpt-4o-mini")

    calc_handler_1 = TokenCalcHandler(token_cost_process=TokenProcess())
    calc_handler_2 = TokenCalcHandler(token_cost_process=TokenProcess())

    result1 = llm1.call(
        messages=[{"role": "user", "content": "Hello, world!"}],
        callbacks=[calc_handler_1],
    )
    print("result1:", result1)
    usage_metrics_1 = calc_handler_1.token_cost_process.get_summary()
    print("usage_metrics_1:", usage_metrics_1)

    result2 = llm2.call(
        messages=[{"role": "user", "content": "Hello, world from another agent!"}],
        callbacks=[calc_handler_2],
    )
    sleep(5)
    print("result2:", result2)
    usage_metrics_2 = calc_handler_2.token_cost_process.get_summary()
    print("usage_metrics_2:", usage_metrics_2)

    # The first handler should not have been updated
    assert usage_metrics_1.successful_requests == 1
    assert usage_metrics_2.successful_requests == 1
    assert usage_metrics_1 == calc_handler_1.token_cost_process.get_summary()
