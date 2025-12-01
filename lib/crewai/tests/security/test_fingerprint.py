"""Test for the Fingerprint class."""

import json
import uuid
from datetime import datetime, timedelta

import pytest
from crewai.security import Fingerprint


def test_fingerprint_creation_with_defaults():
    """Test creating a Fingerprint with default values."""
    fingerprint = Fingerprint()

    # Check that a UUID was generated
    assert fingerprint.uuid_str is not None
    # Check that it's a valid UUID
    uuid_obj = uuid.UUID(fingerprint.uuid_str)
    assert isinstance(uuid_obj, uuid.UUID)

    # Check that creation time was set
    assert isinstance(fingerprint.created_at, datetime)

    # Check that metadata is an empty dict
    assert fingerprint.metadata == {}


def test_fingerprint_creation_with_metadata():
    """Test creating a Fingerprint with custom metadata only."""
    metadata = {"version": "1.0", "author": "Test Author"}

    fingerprint = Fingerprint(metadata=metadata)

    # UUID and created_at should be auto-generated
    assert fingerprint.uuid_str is not None
    assert isinstance(fingerprint.created_at, datetime)
    # Only metadata should be settable
    assert fingerprint.metadata == metadata


def test_fingerprint_uuid_cannot_be_set():
    """Test that uuid_str cannot be manually set."""
    original_uuid = "b723c6ff-95de-5e87-860b-467b72282bd8"

    # Attempt to set uuid_str
    fingerprint = Fingerprint(uuid_str=original_uuid)

    # UUID should be generated, not set to our value
    assert fingerprint.uuid_str != original_uuid
    assert uuid.UUID(fingerprint.uuid_str)  # Should be a valid UUID


def test_fingerprint_created_at_cannot_be_set():
    """Test that created_at cannot be manually set."""
    original_time = datetime.now() - timedelta(days=1)

    # Attempt to set created_at
    fingerprint = Fingerprint(created_at=original_time)

    # created_at should be auto-generated, not set to our value
    assert fingerprint.created_at != original_time
    assert fingerprint.created_at > original_time  # Should be more recent


def test_fingerprint_uuid_property():
    """Test the uuid property returns a UUID object."""
    fingerprint = Fingerprint()

    assert isinstance(fingerprint.uuid, uuid.UUID)
    assert str(fingerprint.uuid) == fingerprint.uuid_str


def test_fingerprint_deterministic_generation():
    """Test that the same seed string always generates the same fingerprint using generate method."""
    seed = "test-seed"

    # Use the generate method which supports deterministic generation
    fingerprint1 = Fingerprint.generate(seed)
    fingerprint2 = Fingerprint.generate(seed)

    assert fingerprint1.uuid_str == fingerprint2.uuid_str

    # Also test with _generate_uuid method directly
    uuid_str1 = Fingerprint._generate_uuid(seed)
    uuid_str2 = Fingerprint._generate_uuid(seed)
    assert uuid_str1 == uuid_str2


def test_fingerprint_generate_classmethod():
    """Test the generate class method."""
    # Without seed
    fingerprint1 = Fingerprint.generate()
    assert isinstance(fingerprint1, Fingerprint)

    # With seed
    seed = "test-seed"
    metadata = {"version": "1.0"}
    fingerprint2 = Fingerprint.generate(seed, metadata)

    assert isinstance(fingerprint2, Fingerprint)
    assert fingerprint2.metadata == metadata

    # Same seed should generate same UUID
    fingerprint3 = Fingerprint.generate(seed)
    assert fingerprint2.uuid_str == fingerprint3.uuid_str


def test_fingerprint_string_representation():
    """Test the string representation of Fingerprint."""
    fingerprint = Fingerprint()
    uuid_str = fingerprint.uuid_str

    string_repr = str(fingerprint)
    assert uuid_str in string_repr


def test_fingerprint_equality():
    """Test fingerprint equality comparison."""
    # Using generate with the same seed to get consistent UUIDs
    seed = "test-equality"

    fingerprint1 = Fingerprint.generate(seed)
    fingerprint2 = Fingerprint.generate(seed)
    fingerprint3 = Fingerprint()

    assert fingerprint1 == fingerprint2
    assert fingerprint1 != fingerprint3


def test_fingerprint_hash():
    """Test that fingerprints can be used as dictionary keys."""
    # Using generate with the same seed to get consistent UUIDs
    seed = "test-hash"

    fingerprint1 = Fingerprint.generate(seed)
    fingerprint2 = Fingerprint.generate(seed)

    # Hash should be consistent for same UUID
    assert hash(fingerprint1) == hash(fingerprint2)

    # Can be used as dict keys
    fingerprint_dict = {fingerprint1: "value"}
    assert fingerprint_dict[fingerprint2] == "value"


def test_fingerprint_to_dict():
    """Test converting fingerprint to dictionary."""
    metadata = {"version": "1.0"}
    fingerprint = Fingerprint(metadata=metadata)

    uuid_str = fingerprint.uuid_str
    created_at = fingerprint.created_at

    fingerprint_dict = fingerprint.to_dict()

    assert fingerprint_dict["uuid_str"] == uuid_str
    assert fingerprint_dict["created_at"] == created_at.isoformat()
    assert fingerprint_dict["metadata"] == metadata


def test_fingerprint_from_dict():
    """Test creating fingerprint from dictionary."""
    uuid_str = "b723c6ff-95de-5e87-860b-467b72282bd8"
    created_at = datetime.now()
    created_at_iso = created_at.isoformat()
    metadata = {"version": "1.0"}

    fingerprint_dict = {
        "uuid_str": uuid_str,
        "created_at": created_at_iso,
        "metadata": metadata,
    }

    fingerprint = Fingerprint.from_dict(fingerprint_dict)

    assert fingerprint.uuid_str == uuid_str
    assert fingerprint.created_at.isoformat() == created_at_iso
    assert fingerprint.metadata == metadata


def test_fingerprint_json_serialization():
    """Test that Fingerprint can be JSON serialized and deserialized."""
    # Create a fingerprint, get its values
    metadata = {"version": "1.0"}
    fingerprint = Fingerprint(metadata=metadata)

    uuid_str = fingerprint.uuid_str
    created_at = fingerprint.created_at

    # Convert to dict and then JSON
    fingerprint_dict = fingerprint.to_dict()
    json_str = json.dumps(fingerprint_dict)

    # Parse JSON and create new fingerprint
    parsed_dict = json.loads(json_str)
    new_fingerprint = Fingerprint.from_dict(parsed_dict)

    assert new_fingerprint.uuid_str == uuid_str
    assert new_fingerprint.created_at.isoformat() == created_at.isoformat()
    assert new_fingerprint.metadata == metadata


def test_invalid_uuid_str():
    """Test handling of invalid UUID strings."""
    uuid_str = "not-a-valid-uuid"
    created_at = datetime.now().isoformat()

    fingerprint_dict = {"uuid_str": uuid_str, "created_at": created_at, "metadata": {}}

    # The Fingerprint.from_dict method accepts even invalid UUIDs
    # This seems to be the current behavior
    fingerprint = Fingerprint.from_dict(fingerprint_dict)

    # Verify it uses the provided UUID string, even if invalid
    # This might not be ideal behavior, but it's the current implementation
    assert fingerprint.uuid_str == uuid_str

    # But this will raise an exception when we try to access the uuid property
    with pytest.raises(ValueError):
        uuid_obj = fingerprint.uuid


def test_fingerprint_metadata_mutation():
    """Test that metadata can be modified after fingerprint creation."""
    # Create a fingerprint with initial metadata
    initial_metadata = {"version": "1.0", "status": "draft"}
    fingerprint = Fingerprint(metadata=initial_metadata)

    # Verify initial metadata
    assert fingerprint.metadata == initial_metadata

    # Modify the metadata
    fingerprint.metadata["status"] = "published"
    fingerprint.metadata["author"] = "Test Author"

    # Verify the modifications
    expected_metadata = {
        "version": "1.0",
        "status": "published",
        "author": "Test Author",
    }
    assert fingerprint.metadata == expected_metadata

    # Make sure the UUID and creation time remain unchanged
    uuid_str = fingerprint.uuid_str
    created_at = fingerprint.created_at

    # Completely replace the metadata
    new_metadata = {"version": "2.0", "environment": "production"}
    fingerprint.metadata = new_metadata

    # Verify the replacement
    assert fingerprint.metadata == new_metadata

    # Ensure immutable fields remain unchanged
    assert fingerprint.uuid_str == uuid_str
    assert fingerprint.created_at == created_at
