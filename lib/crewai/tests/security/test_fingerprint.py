"""Test for the Fingerprint class."""

from datetime import datetime, timedelta
import json
import uuid

from crewai.security import Fingerprint
import pytest


def test_fingerprint_creation_with_defaults():
    """Test creating a Fingerprint with default values."""
    fingerprint = Fingerprint()

    assert fingerprint.uuid_str is not None
    uuid_obj = uuid.UUID(fingerprint.uuid_str)
    assert isinstance(uuid_obj, uuid.UUID)

    assert isinstance(fingerprint.created_at, datetime)

    assert fingerprint.metadata == {}


def test_fingerprint_creation_with_metadata():
    """Test creating a Fingerprint with custom metadata only."""
    metadata = {"version": "1.0", "author": "Test Author"}

    fingerprint = Fingerprint(metadata=metadata)

    assert fingerprint.uuid_str is not None
    assert isinstance(fingerprint.created_at, datetime)
    assert fingerprint.metadata == metadata


def test_fingerprint_uuid_cannot_be_set():
    """Test that uuid_str cannot be manually set."""
    original_uuid = "b723c6ff-95de-5e87-860b-467b72282bd8"

    fingerprint = Fingerprint(uuid_str=original_uuid)

    assert fingerprint.uuid_str != original_uuid
    assert uuid.UUID(fingerprint.uuid_str)


def test_fingerprint_created_at_cannot_be_set():
    """Test that created_at cannot be manually set."""
    original_time = datetime.now() - timedelta(days=1)

    fingerprint = Fingerprint(created_at=original_time)

    assert fingerprint.created_at != original_time
    assert fingerprint.created_at > original_time


def test_fingerprint_uuid_property():
    """Test the uuid property returns a UUID object."""
    fingerprint = Fingerprint()

    assert isinstance(fingerprint.uuid, uuid.UUID)
    assert str(fingerprint.uuid) == fingerprint.uuid_str


def test_fingerprint_deterministic_generation():
    """Test that the same seed string always generates the same fingerprint using generate method."""
    seed = "test-seed"

    fingerprint1 = Fingerprint.generate(seed)
    fingerprint2 = Fingerprint.generate(seed)

    assert fingerprint1.uuid_str == fingerprint2.uuid_str

    uuid_str1 = Fingerprint._generate_uuid(seed)
    uuid_str2 = Fingerprint._generate_uuid(seed)
    assert uuid_str1 == uuid_str2


def test_fingerprint_generate_classmethod():
    """Test the generate class method."""
    fingerprint1 = Fingerprint.generate()
    assert isinstance(fingerprint1, Fingerprint)

    seed = "test-seed"
    metadata = {"version": "1.0"}
    fingerprint2 = Fingerprint.generate(seed, metadata)

    assert isinstance(fingerprint2, Fingerprint)
    assert fingerprint2.metadata == metadata

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
    seed = "test-equality"

    fingerprint1 = Fingerprint.generate(seed)
    fingerprint2 = Fingerprint.generate(seed)
    fingerprint3 = Fingerprint()

    assert fingerprint1 == fingerprint2
    assert fingerprint1 != fingerprint3


def test_fingerprint_hash():
    """Test that fingerprints can be used as dictionary keys."""
    seed = "test-hash"

    fingerprint1 = Fingerprint.generate(seed)
    fingerprint2 = Fingerprint.generate(seed)

    assert hash(fingerprint1) == hash(fingerprint2)

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
    metadata = {"version": "1.0"}
    fingerprint = Fingerprint(metadata=metadata)

    uuid_str = fingerprint.uuid_str
    created_at = fingerprint.created_at

    fingerprint_dict = fingerprint.to_dict()
    json_str = json.dumps(fingerprint_dict)

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

    # This seems to be the current behavior
    fingerprint = Fingerprint.from_dict(fingerprint_dict)

    # This might not be ideal behavior, but it's the current implementation
    assert fingerprint.uuid_str == uuid_str

    # But this will raise an exception when we try to access the uuid property
    with pytest.raises(ValueError):
        uuid_obj = fingerprint.uuid


def test_fingerprint_metadata_mutation():
    """Test that metadata can be modified after fingerprint creation."""
    initial_metadata = {"version": "1.0", "status": "draft"}
    fingerprint = Fingerprint(metadata=initial_metadata)

    assert fingerprint.metadata == initial_metadata

    fingerprint.metadata["status"] = "published"
    fingerprint.metadata["author"] = "Test Author"

    expected_metadata = {
        "version": "1.0",
        "status": "published",
        "author": "Test Author",
    }
    assert fingerprint.metadata == expected_metadata

    uuid_str = fingerprint.uuid_str
    created_at = fingerprint.created_at

    # Completely replace the metadata
    new_metadata = {"version": "2.0", "environment": "production"}
    fingerprint.metadata = new_metadata

    assert fingerprint.metadata == new_metadata

    assert fingerprint.uuid_str == uuid_str
    assert fingerprint.created_at == created_at
