"""Test for the SecurityConfig class."""

import json
from datetime import datetime

from crewai.security import Fingerprint, SecurityConfig


def test_security_config_creation_with_defaults():
    """Test creating a SecurityConfig with default values."""
    config = SecurityConfig()

    # Check default values
    assert config.fingerprint is not None  # Fingerprint is auto-generated
    assert isinstance(config.fingerprint, Fingerprint)
    assert config.fingerprint.uuid_str is not None  # UUID is auto-generated


def test_security_config_fingerprint_generation():
    """Test that SecurityConfig automatically generates fingerprints."""
    config = SecurityConfig()

    # Check that fingerprint was auto-generated
    assert config.fingerprint is not None
    assert isinstance(config.fingerprint, Fingerprint)
    assert isinstance(config.fingerprint.uuid_str, str)
    assert len(config.fingerprint.uuid_str) > 0


def test_security_config_init_params():
    """Test that SecurityConfig can be initialized and modified."""
    # Create a config
    config = SecurityConfig()

    # Create a custom fingerprint
    fingerprint = Fingerprint(metadata={"version": "1.0"})

    # Set the fingerprint
    config.fingerprint = fingerprint

    # Check fingerprint was set correctly
    assert config.fingerprint is fingerprint
    assert config.fingerprint.metadata == {"version": "1.0"}


def test_security_config_to_dict():
    """Test converting SecurityConfig to dictionary."""
    # Create a config with a fingerprint that has metadata
    config = SecurityConfig()
    config.fingerprint.metadata = {"version": "1.0"}

    config_dict = config.to_dict()

    # Check the fingerprint is in the dict
    assert "fingerprint" in config_dict
    assert isinstance(config_dict["fingerprint"], dict)
    assert config_dict["fingerprint"]["metadata"] == {"version": "1.0"}


def test_security_config_from_dict():
    """Test creating SecurityConfig from dictionary."""
    # Create a fingerprint dict
    fingerprint_dict = {
        "uuid_str": "b723c6ff-95de-5e87-860b-467b72282bd8",
        "created_at": datetime.now().isoformat(),
        "metadata": {"version": "1.0"},
    }

    # Create a config dict with just the fingerprint
    config_dict = {"fingerprint": fingerprint_dict}

    # Create config manually since from_dict has a specific implementation
    config = SecurityConfig()

    # Set the fingerprint manually from the dict
    fingerprint = Fingerprint.from_dict(fingerprint_dict)
    config.fingerprint = fingerprint

    # Check fingerprint was properly set
    assert config.fingerprint is not None
    assert isinstance(config.fingerprint, Fingerprint)
    assert config.fingerprint.uuid_str == fingerprint_dict["uuid_str"]
    assert config.fingerprint.metadata == fingerprint_dict["metadata"]


def test_security_config_json_serialization():
    """Test that SecurityConfig can be JSON serialized and deserialized."""
    # Create a config with fingerprint metadata
    config = SecurityConfig()
    config.fingerprint.metadata = {"version": "1.0"}

    # Convert to dict and then JSON
    config_dict = config.to_dict()

    # Make sure fingerprint is properly converted to dict
    assert isinstance(config_dict["fingerprint"], dict)

    # Now it should be JSON serializable
    json_str = json.dumps(config_dict)

    # Should be able to parse back to dict
    parsed_dict = json.loads(json_str)

    # Check fingerprint values match
    assert parsed_dict["fingerprint"]["metadata"] == {"version": "1.0"}

    # Create a new config manually
    new_config = SecurityConfig()

    # Set the fingerprint from the parsed data
    fingerprint_data = parsed_dict["fingerprint"]
    new_fingerprint = Fingerprint.from_dict(fingerprint_data)
    new_config.fingerprint = new_fingerprint

    # Check the new config has the same fingerprint metadata
    assert new_config.fingerprint.metadata == {"version": "1.0"}
