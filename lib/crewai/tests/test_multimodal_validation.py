import os

import pytest

from crewai import LLM, Agent, Crew, Task


@pytest.mark.skip(reason="Only run manually with valid API keys")
def test_multimodal_agent_with_image_url():
    """
    Test that a multimodal agent can process images without validation errors.
    This test reproduces the scenario from issue #2475.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    llm = LLM(
        model="openai/gpt-4o",  # model with vision capabilities
        api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    expert_analyst = Agent(
        role="Visual Quality Inspector",
        goal="Perform detailed quality analysis of product images",
        backstory="Senior quality control expert with expertise in visual inspection",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        multimodal=True
    )

    inspection_task = Task(
        description="""
        Analyze the product image at https://www.us.maguireshoes.com/collections/spring-25/products/lucena-black-boot with focus on:
        1. Quality of materials
        2. Manufacturing defects
        3. Compliance with standards
        Provide a detailed report highlighting any issues found.
        """,
        expected_output="A detailed report highlighting any issues found",
        agent=expert_analyst
    )

    crew = Crew(agents=[expert_analyst], tasks=[inspection_task])
