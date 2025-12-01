"""Tests for import utilities."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from crewai.utilities.import_utils import (
    OptionalDependencyError,
    import_and_validate_definition,
    require,
    validate_import_path,
)


class TestRequire:
    """Test the require function."""

    def test_require_existing_module(self):
        """Test requiring a module that exists."""
        module = require("json", purpose="testing")
        assert module.__name__ == "json"

    def test_require_missing_module(self):
        """Test requiring a module that doesn't exist."""
        with pytest.raises(OptionalDependencyError) as exc_info:
            require("nonexistent_module_xyz", purpose="testing missing module")

        error_msg = str(exc_info.value)
        assert (
            "testing missing module requires the optional dependency 'nonexistent_module_xyz'"
            in error_msg
        )
        assert "uv add nonexistent_module_xyz" in error_msg

    def test_require_with_import_error(self):
        """Test that ImportError is properly chained."""
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module import failed")

            with pytest.raises(OptionalDependencyError) as exc_info:
                require("some_module", purpose="testing error handling")

            assert isinstance(exc_info.value.__cause__, ImportError)
            assert str(exc_info.value.__cause__) == "Module import failed"

    def test_optional_dependency_error_is_import_error(self):
        """Test that OptionalDependencyError is a subclass of ImportError."""
        assert issubclass(OptionalDependencyError, ImportError)

    def test_require_with_attr(self):
        """Test requiring a specific attribute from a module."""
        loads = require("json", purpose="testing", attr="loads")
        import json

        assert loads == json.loads

    def test_require_with_nonexistent_attr(self):
        """Test requiring a nonexistent attribute raises AttributeError."""
        with pytest.raises(AttributeError) as exc_info:
            require("json", purpose="testing", attr="nonexistent_attr")

        assert "Module 'json' has no attribute 'nonexistent_attr'" in str(
            exc_info.value
        )

    def test_require_extracts_package_name(self):
        """Test that require correctly extracts package name from module path."""
        with pytest.raises(OptionalDependencyError) as exc_info:
            require("some.nested.module.path", purpose="testing")

        error_msg = str(exc_info.value)
        assert "uv add some" in error_msg


class TestValidateImportPath:
    """Test the validate_import_path function."""

    def test_validate_import_path_success(self):
        """Test successful import of a class."""
        result = validate_import_path("json.JSONDecoder")
        import json

        assert result == json.JSONDecoder

    def test_validate_import_path_malformed_no_module(self):
        """Test validation with no module path."""
        with pytest.raises(ValueError) as exc_info:
            validate_import_path("ClassName")

        assert "import_path 'ClassName' must be of the form 'module.ClassName'" in str(
            exc_info.value
        )

    def test_validate_import_path_empty_string(self):
        """Test validation with empty string."""
        with pytest.raises(ValueError) as exc_info:
            validate_import_path("")

        assert "import_path '' must be of the form 'module.ClassName'" in str(
            exc_info.value
        )

    def test_validate_import_path_module_not_found(self):
        """Test validation with non-existent module."""
        with pytest.raises(ValueError) as exc_info:
            validate_import_path("nonexistent_module.ClassName")

        error_msg = str(exc_info.value)
        assert "Package 'nonexistent_module' could not be imported" in error_msg
        assert "uv add nonexistent_module" in error_msg

    def test_validate_import_path_attribute_not_found(self):
        """Test validation when attribute doesn't exist in module."""
        with pytest.raises(ValueError) as exc_info:
            validate_import_path("json.NonExistentClass")

        assert "Attribute 'NonExistentClass' not found in module 'json'" in str(
            exc_info.value
        )

    def test_validate_import_path_nested_module(self):
        """Test validation with nested module path."""
        result = validate_import_path("unittest.mock.MagicMock")
        from unittest.mock import MagicMock

        assert result == MagicMock

    def test_validate_import_path_extracts_package_name(self):
        """Test that package name is correctly extracted for error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_import_path("some.nested.module.path.ClassName")

        error_msg = str(exc_info.value)
        assert "Package 'some' could not be imported" in error_msg
        assert "uv add some" in error_msg


class TestImportAndValidateDefinition:
    """Test the import_and_validate_definition function."""

    def test_import_and_validate_definition_success(self):
        """Test successful import through Pydantic adapter."""
        result = import_and_validate_definition("json.JSONEncoder")
        import json

        assert result == json.JSONEncoder

    def test_import_and_validate_definition_with_function(self):
        """Test importing a function instead of a class."""
        result = import_and_validate_definition("json.loads")
        import json

        assert result == json.loads

    def test_import_and_validate_definition_invalid(self):
        """Test that invalid paths raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            import_and_validate_definition("InvalidPath")

        assert "must be of the form 'module.ClassName'" in str(exc_info.value)

    def test_import_and_validate_definition_module_error(self):
        """Test error handling for missing modules."""
        with pytest.raises(ValueError) as exc_info:
            import_and_validate_definition("missing_package.SomeClass")

        error_msg = str(exc_info.value)
        assert "Package 'missing_package' could not be imported" in error_msg
        assert "uv add missing_package" in error_msg

    def test_import_and_validate_definition_attribute_error(self):
        """Test error handling for missing attributes."""
        with pytest.raises(ValueError) as exc_info:
            import_and_validate_definition("json.MissingClass")

        assert "Attribute 'MissingClass' not found in module 'json'" in str(
            exc_info.value
        )

    def test_import_and_validate_definition_with_mock(self):
        """Test that mocked modules work correctly."""
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.MockClass = mock_class

        with patch.dict(sys.modules, {"mocked_module": mock_module}):
            result = import_and_validate_definition("mocked_module.MockClass")
            assert result == mock_class
