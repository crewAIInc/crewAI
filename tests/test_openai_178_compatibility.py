import os

import pytest

from crewai import LLM, Agent, Crew, Task, TaskOutput


@pytest.mark.skip(reason="Only run manually with valid API keys")
def test_openai_178_compatibility_with_multimodal():
    """
    Test that CrewAI works with OpenAI 1.78.0 and multi-image input support.
    This test verifies the fix for issue #2910.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    llm = LLM(
        model="openai/gpt-4o",  # model with vision capabilities
        api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    visual_agent = Agent(
        role="Multi-Image Analyst",
        goal="Analyze multiple images and provide comprehensive reports",
        backstory="Expert in visual analysis capable of processing multiple images simultaneously",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        multimodal=True
    )

    analysis_task = Task(
        description="""
        Analyze these product images:
        1. https://www.us.maguireshoes.com/cdn/shop/files/FW24-Edito-Lucena-Distressed-01_1920x.jpg?v=1736371244
        2. https://example.com/sample-image.jpg
        
        Provide a comparative analysis focusing on design elements and quality indicators.
        """,
        expected_output="A comparative analysis of the provided images",
        agent=visual_agent
    )

    crew = Crew(agents=[visual_agent], tasks=[analysis_task])
    result = crew.kickoff()

    assert result is not None
    assert len(result.tasks_output) == 1
    task_output = result.tasks_output[0]
    assert isinstance(task_output, TaskOutput)
    assert len(task_output.raw) > 0
