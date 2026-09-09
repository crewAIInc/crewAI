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


@patch("crewai_tools.tools.vision_tool.vision_tool.LLM")
def test_reading_llm_property_does_not_override_complexity_level(mock_llm_cls):
    """A cached ``self._llm`` (side effect of reading ``llm``) must not win.

    Regression for the bug where any populated ``self._llm`` was treated as an
    explicitly supplied LLM, causing every subsequent call to reuse the cached
    model regardless of ``complexity_level``.
    """
    mock_llm_cls.return_value = _llm_mock()

    tool = VisionTool()
    _ = tool.llm  # caches the medium-tier model as a side effect
    mock_llm_cls.assert_called_once_with(model=DEFAULT_MODEL, stop=["STOP", "END"])

    mock_llm_cls.reset_mock()
    tool._run(image_path_url=IMAGE_URL, complexity_level="hard")

    mock_llm_cls.assert_called_once_with(
        model=COMPLEXITY_MODEL_MAP["hard"], stop=["STOP", "END"]
    )


@patch("crewai_tools.tools.vision_tool.vision_tool.LLM")
def test_same_instance_resolves_model_per_call(mock_llm_cls):
    """One instance across two calls resolves each call's model independently."""
    mock_llm_cls.side_effect = lambda **kwargs: _llm_mock()

    tool = VisionTool()
    tool._run(image_path_url=IMAGE_URL, complexity_level="easy")
    tool._run(image_path_url=IMAGE_URL, complexity_level="hard")

    used_models = [call.kwargs["model"] for call in mock_llm_cls.call_args_list]
    assert used_models == [
        COMPLEXITY_MODEL_MAP["easy"],
        COMPLEXITY_MODEL_MAP["hard"],
    ]


@patch("crewai_tools.tools.vision_tool.vision_tool.LLM")
def test_complexity_llm_is_cached_per_level(mock_llm_cls):
    """Repeated calls with the same complexity level reuse a single LLM."""
    mock_llm_cls.return_value = _llm_mock()

    tool = VisionTool()
    tool._run(image_path_url=IMAGE_URL, complexity_level="hard")
    tool._run(image_path_url=IMAGE_URL, complexity_level="hard")

    mock_llm_cls.assert_called_once_with(
        model=COMPLEXITY_MODEL_MAP["hard"], stop=["STOP", "END"]
    )


def test_invalid_complexity_level_returns_error_string():
    """An unexpected complexity level fails validation and is reported as an error.

    ``ImagePromptSchema``'s ``Literal`` validation raises, which ``_run`` catches
    and surfaces as a generic error string rather than propagating.
    """
    tool = VisionTool()

    result = tool._run(image_path_url=IMAGE_URL, complexity_level="very hard")

    assert result.startswith("An error occurred")
