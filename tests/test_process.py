"""Test Process enum functionality."""

import pytest
from enum import Enum

from crewai.process import Process


class TestProcess:
    """Test suite for Process enum."""

    def test_process_is_enum(self):
        """Test that Process is an Enum."""
        assert issubclass(Process, Enum)
        assert issubclass(Process, str)

    def test_process_values(self):
        """Test that Process enum has correct values."""
        assert Process.sequential.value == "sequential"
        assert Process.hierarchical.value == "hierarchical"

    def test_process_string_representation(self):
        """Test that Process enum values can be used as strings.

        Note: Even though Process subclasses str, str() returns the qualified name
        (e.g., "Process.sequential") while the enum member itself equals its value
        (e.g., Process.sequential == "sequential").
        """
        # str() returns qualified name for better debugging
        assert str(Process.sequential) == "Process.sequential"
        assert str(Process.hierarchical) == "Process.hierarchical"

        # f-string formatting behaves the same as str()
        assert f"{Process.sequential}" == "Process.sequential"
        assert f"{Process.hierarchical}" == "Process.hierarchical"

        # The enum member itself equals its string value (because it subclasses str)
        assert Process.sequential == "sequential"
        assert Process.hierarchical == "hierarchical"

        # .value gives the actual string value
        assert Process.sequential.value == "sequential"
        assert Process.hierarchical.value == "hierarchical"

    def test_process_comparison(self):
        """Test Process enum comparison."""
        assert Process.sequential == Process.sequential
        assert Process.sequential != Process.hierarchical
        assert Process.sequential == "sequential"
        assert Process.hierarchical == "hierarchical"

    def test_process_iteration(self):
        """Test that we can iterate over Process values."""
        processes = list(Process)
        assert len(processes) == 2
        assert Process.sequential in processes
        assert Process.hierarchical in processes

    def test_process_membership(self):
        """Test membership checks for Process enum."""
        assert "sequential" in Process._value2member_map_
        assert "hierarchical" in Process._value2member_map_
        assert "invalid" not in Process._value2member_map_

    def test_process_from_string(self):
        """Test creating Process from string value."""
        assert Process("sequential") == Process.sequential
        assert Process("hierarchical") == Process.hierarchical

    def test_process_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Process("invalid_process")
        assert "invalid_process" in str(exc_info.value)

    def test_process_name_attribute(self):
        """Test the name attribute of Process enum members."""
        assert Process.sequential.name == "sequential"
        assert Process.hierarchical.name == "hierarchical"

    def test_process_immutability(self):
        """Test that Process enum values cannot be modified."""
        with pytest.raises(AttributeError):
            Process.sequential = "modified"

    def test_process_hash(self):
        """Test that Process enum members are hashable."""
        process_set = {Process.sequential, Process.hierarchical}
        assert len(process_set) == 2
        assert Process.sequential in process_set

    def test_process_in_dict(self):
        """Test using Process enum as dictionary keys."""
        process_config = {
            Process.sequential: {"order": "linear"},
            Process.hierarchical: {"order": "tree"}
        }
        assert process_config[Process.sequential] == {"order": "linear"}
        assert process_config[Process.hierarchical] == {"order": "tree"}

    def test_process_string_enum_behavior(self):
        """Test that Process behaves correctly as a string enum.

        String enums (Enum subclassing str) have special behavior:
        - The member itself can be compared directly to strings
        - Can be used anywhere a string is expected
        - str() returns qualified name for clarity
        """
        # Can be used in string operations
        assert Process.sequential.upper() == "SEQUENTIAL"
        assert Process.hierarchical.lower() == "hierarchical"

        # Can be used in string methods
        assert Process.sequential.startswith("seq")
        assert Process.hierarchical.endswith("cal")

        # Length and indexing work
        assert len(Process.sequential) == len("sequential")
        assert Process.sequential[0] == "s"

        # Format string behavior (note: %s and {} use str() which returns qualified name)
        assert "Mode: %s" % Process.sequential == "Mode: Process.sequential"
        assert "Mode: {}".format(Process.sequential) == "Mode: Process.sequential"

        # To get the value in format strings, use .value explicitly
        assert "Mode: %s" % Process.sequential.value == "Mode: sequential"
        assert "Mode: {}".format(Process.sequential.value) == "Mode: sequential"