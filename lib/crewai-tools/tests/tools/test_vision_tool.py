from unittest.mock import MagicMock, patch

from crewai_tools.tools.vision_tool.vision_tool import (
    COMPLEXITY_MODEL_MAP,
    DEFAULT_MODEL,
    VisionTool,
)
import pytest

IMAGE_URL = "http://example.com/image.png"


def _llm_mock(return_value: str = "described") -> MagicMock:
    llm = MagicMock()
    llm.call.return_value = return_value
    return llm


def test_explicit_llm_is_used_over_complexity_map():
    """An explicitly provided LLM takes precedence over the complexity map."""
    llm = _llm_mock("from explicit llm")

    tool = VisionTool(llm=llm)
    result = tool._run(image_path_url=IMAGE_URL, complexity_level="hard")

    assert result == "from explicit llm"
    llm.call.assert_called_once()


@patch("crewai_tools.tools.vision_tool.vision_tool.LLM")
def test_explicit_model_overrides_complexity_map(mock_llm_cls):
    """VisionTool(model="custom") must use "custom", not the complexity map."""
    mock_llm_cls.return_value = _llm_mock()

    tool = VisionTool(model="custom-model")
    tool._run(image_path_url=IMAGE_URL, complexity_level="hard")

    mock_llm_cls.assert_called_once_with(model="custom-model", stop=["STOP", "END"])


@patch("crewai_tools.tools.vision_tool.vision_tool.LLM")
def test_model_setter_marks_model_explicit(mock_llm_cls):
    """Assigning through the model setter also overrides the complexity map."""
    mock_llm_cls.return_value = _llm_mock()

    tool = VisionTool()
    tool.model = "setter-model"
    tool._run(image_path_url=IMAGE_URL, complexity_level="easy")

    mock_llm_cls.assert_called_once_with(model="setter-model", stop=["STOP", "END"])


@pytest.mark.parametrize(("complexity_level", "expected_model"), COMPLEXITY_MODEL_MAP.items())
@patch("crewai_tools.tools.vision_tool.vision_tool.LLM")
def test_complexity_level_selects_expected_model(
    mock_llm_cls, complexity_level, expected_model
):
    """Each complexity level maps to its corresponding model when none is set."""
    mock_llm_cls.return_value = _llm_mock()

    tool = VisionTool()
    tool._run(image_path_url=IMAGE_URL, complexity_level=complexity_level)

    mock_llm_cls.assert_called_once_with(model=expected_model, stop=["STOP", "END"])


@patch("crewai_tools.tools.vision_tool.vision_tool.LLM")
def test_default_uses_medium_complexity_model(mock_llm_cls):
    """Without a model, llm, or complexity level, the default (medium) is used."""
    mock_llm_cls.return_value = _llm_mock()

    tool = VisionTool()
    tool._run(image_path_url=IMAGE_URL)

    mock_llm_cls.assert_called_once_with(model=DEFAULT_MODEL, stop=["STOP", "END"])
