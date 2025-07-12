import pytest
from crewai.tools.agent_tools.analyze_image_tool import AnalyzeImageTool

def analyze_image_with_tool(image_url: str):
    tool = AnalyzeImageTool()
    try:
        result = tool._run(
            image_url=image_url,
            action="Analyze and describe the content of the image",
        )
        return result
    except Exception as e:
        return None




@pytest.mark.vcr
def test_analyze_image_url_runs():
    # This test checks if the function runs and returns a string (could be empty if API key or model is not set)
    image_url = "https://images.unsplash.com/photo-1554866585-cd94860890b7?q=80&w=1065&auto=format"
    result = analyze_image_with_tool(image_url)
    assert isinstance(result, str)

    result = result.lower()
    possible_values = ["coca-cola", "coca", "cola"]
    assert any(value in result for value in possible_values)
