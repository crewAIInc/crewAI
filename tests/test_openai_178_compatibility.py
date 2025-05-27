import os

import pytest

from crewai import LLM, Agent, Crew, Task, TaskOutput

TEST_IMAGES = {
    "product_shoe": "https://www.us.maguireshoes.com/cdn/shop/files/FW24-Edito-Lucena-Distressed-01_1920x.jpg?v=1736371244",
    "sample_image": "https://example.com/sample-image.jpg"
}


@pytest.mark.requires_api_key
@pytest.mark.skip(reason="Only run manually with valid API keys")
def test_openai_178_compatibility_with_multimodal():
    """Test CrewAI compatibility with OpenAI 1.78.0 multi-image support."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Test requires OPENAI_API_KEY environment variable")

    llm = LLM(
        model="openai/gpt-4o",  # model with vision capabilities
        api_key=os.getenv("OPENAI_API_KEY"),
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
        1. {product_shoe}
        2. {sample_image}
        
        Provide a comparative analysis focusing on design elements and quality indicators.
        """.format(
            product_shoe=TEST_IMAGES["product_shoe"],
            sample_image=TEST_IMAGES["sample_image"]
        ),
        expected_output="A comparative analysis of the provided images",
        agent=visual_agent
    )

    crew = Crew(agents=[visual_agent], tasks=[analysis_task])
    result = crew.kickoff()

    assert result is not None, "Crew execution returned None"
    assert len(result.tasks_output) == 1, "Expected exactly one task output"
    task_output = result.tasks_output[0]
    assert isinstance(task_output, TaskOutput), f"Expected TaskOutput, got {type(task_output)}"
    assert len(task_output.raw) > 0, "Task output should contain content"
