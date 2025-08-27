"""Test that all public API classes are properly importable."""


def test_task_output_import():
    """Test that TaskOutput can be imported from crewai."""
    from crewai import TaskOutput

    assert TaskOutput is not None


def test_crew_output_import():
    """Test that CrewOutput can be imported from crewai."""
    from crewai import CrewOutput

    assert CrewOutput is not None


def test_onnxruntime_import_and_version():
    """Test that onnxruntime can be imported and is version >= 1.22.1."""
    import onnxruntime
    from packaging import version
    
    assert onnxruntime is not None
    assert version.parse(onnxruntime.__version__) >= version.parse("1.22.1")
