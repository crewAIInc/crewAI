"""Tests for ChromaDBConfig Pydantic V1/V2 compatibility."""

import pytest
from chromadb.config import Settings

from crewai.rag.chromadb.config import ChromaDBConfig, _coerce_settings


class TestCoerceSettings:
    """Test suite for _coerce_settings validator function."""

    def test_coerce_settings_passes_through_settings_instance(self):
        """Test that existing Settings instances are passed through unchanged."""
        settings = Settings(
            persist_directory="./test_db",
            allow_reset=True,
            is_persistent=False,
        )
        result = _coerce_settings(settings)
        assert result is settings
        assert result.persist_directory == "./test_db"
        assert result.allow_reset is True
        assert result.is_persistent is False

    def test_coerce_settings_converts_dict_to_settings(self):
        """Test that dict inputs are converted to Settings instances."""
        settings_dict = {
            "persist_directory": "./my_custom_db",
            "allow_reset": True,
            "is_persistent": True,
        }
        result = _coerce_settings(settings_dict)
        assert isinstance(result, Settings)
        assert result.persist_directory == "./my_custom_db"
        assert result.allow_reset is True
        assert result.is_persistent is True

    def test_coerce_settings_raises_type_error_for_invalid_input(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="settings must be a chromadb.config.Settings"):
            _coerce_settings("invalid_string")

        with pytest.raises(TypeError, match="settings must be a chromadb.config.Settings"):
            _coerce_settings(123)

        with pytest.raises(TypeError, match="settings must be a chromadb.config.Settings"):
            _coerce_settings(["list", "of", "items"])


class TestChromaDBConfigPydanticCompatibility:
    """Test suite for ChromaDBConfig Pydantic V1/V2 compatibility.

    These tests verify the fix for GitHub issue #4095:
    Pydantic V1/V2 Compatibility Crash in RagTool when passing custom ChromaDB Settings.
    """

    def test_chromadb_config_accepts_settings_instance(self):
        """Test that ChromaDBConfig accepts a chromadb.config.Settings instance.

        This is the main regression test for issue #4095 where passing a Settings
        instance would cause: TypeError: BaseModel.validate() takes 2 positional
        arguments but 3 were given
        """
        custom_settings = Settings(
            persist_directory="./my_db",
            allow_reset=True,
            is_persistent=False,
        )
        config = ChromaDBConfig(settings=custom_settings)

        assert config.settings is custom_settings
        assert config.settings.persist_directory == "./my_db"
        assert config.settings.allow_reset is True
        assert config.settings.is_persistent is False

    def test_chromadb_config_accepts_settings_dict(self):
        """Test that ChromaDBConfig accepts a dict for settings and converts it."""
        settings_dict = {
            "persist_directory": "./dict_db",
            "allow_reset": False,
            "is_persistent": True,
        }
        config = ChromaDBConfig(settings=settings_dict)

        assert isinstance(config.settings, Settings)
        assert config.settings.persist_directory == "./dict_db"
        assert config.settings.allow_reset is False
        assert config.settings.is_persistent is True

    def test_chromadb_config_uses_default_settings_when_not_provided(self):
        """Test that ChromaDBConfig uses default settings when none provided."""
        config = ChromaDBConfig()

        assert isinstance(config.settings, Settings)
        assert config.settings.allow_reset is True
        assert config.settings.is_persistent is True

    def test_chromadb_config_with_all_parameters(self):
        """Test ChromaDBConfig with all parameters including custom settings."""
        custom_settings = Settings(
            persist_directory="./full_test_db",
            allow_reset=True,
            is_persistent=True,
        )
        config = ChromaDBConfig(
            tenant="test_tenant",
            database="test_database",
            settings=custom_settings,
            limit=10,
            score_threshold=0.8,
            batch_size=50,
        )

        assert config.tenant == "test_tenant"
        assert config.database == "test_database"
        assert config.settings is custom_settings
        assert config.limit == 10
        assert config.score_threshold == 0.8
        assert config.batch_size == 50

    def test_chromadb_config_provider_is_chromadb(self):
        """Test that provider field is always 'chromadb'."""
        config = ChromaDBConfig()
        assert config.provider == "chromadb"

    def test_chromadb_config_is_frozen(self):
        """Test that ChromaDBConfig is immutable (frozen)."""
        config = ChromaDBConfig()
        with pytest.raises(AttributeError):
            config.tenant = "new_tenant"
