import pytest


def test_basic_import():
    """
    Tests that the crewai package can be imported without raising exceptions.
    This helps catch basic installation and dependency issues, including import
    errors related to Python version compatibility.
    """
    try:
        import crewai
    except ImportError as e:
        pytest.fail(f"Failed to import crewai package: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during crewai import: {e}")
