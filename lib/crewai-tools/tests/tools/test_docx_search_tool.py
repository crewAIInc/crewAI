from typing import Any
from unittest.mock import patch
import pytest
from pydantic import ValidationError

from crewai_tools.tools.docx_search_tool.docx_search_tool import (
    DOCXSearchTool,
    DOCXSearchToolSchema,
    FixedDOCXSearchToolSchema,
)


def test_fixed_docx_search_tool_schema() -> None:
    """FixedDOCXSearchToolSchema should only require search_query."""
    data = FixedDOCXSearchToolSchema.model_validate({"search_query": "quarterly revenue"})
    assert data.search_query == "quarterly revenue"

    # Missing search_query should fail
    with pytest.raises(ValidationError):
        FixedDOCXSearchToolSchema.model_validate({})


def test_docx_search_tool_schema() -> None:
    """DOCXSearchToolSchema should require both docx and search_query."""
    data = DOCXSearchToolSchema.model_validate({
        "docx": "path/to/report.docx",
        "search_query": "quarterly revenue",
    })
    assert data.docx == "path/to/report.docx"
    assert data.search_query == "quarterly revenue"

    # Missing docx should fail
    with pytest.raises(ValidationError):
        DOCXSearchToolSchema.model_validate({"search_query": "quarterly revenue"})

    # Missing search_query should fail
    with pytest.raises(ValidationError):
        DOCXSearchToolSchema.model_validate({"docx": "path/to/report.docx"})


def test_docx_search_tool_initialization_schemas() -> None:
    """Verify tool args_schema matches initialization mode without requiring docx in fixed mode."""
    with patch.object(DOCXSearchTool, "add"):
        # When initialized with a docx file (fixed mode), args_schema should be FixedDOCXSearchToolSchema
        fixed_tool = DOCXSearchTool(docx="sample.docx")
        assert fixed_tool.args_schema == FixedDOCXSearchToolSchema

        # Validating args without docx must succeed for fixed tool
        validated: Any = fixed_tool.args_schema.model_validate({"search_query": "find revenue"})
        assert validated.search_query == "find revenue"
        assert not hasattr(validated, "docx")

        # When initialized without a docx file, args_schema should be DOCXSearchToolSchema
        dynamic_tool = DOCXSearchTool()
        assert dynamic_tool.args_schema == DOCXSearchToolSchema
