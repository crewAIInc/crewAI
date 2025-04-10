"""Test that all public API classes are properly importable."""


def test_task_output_import():
    """Test that TaskOutput can be imported from crewai."""
    from crewai import TaskOutput
    
    assert TaskOutput is not None
    
    
def test_crew_output_import():
    """Test that CrewOutput can be imported from crewai."""
    from crewai import CrewOutput
    
    assert CrewOutput is not None


def test_memory_imports():
    """Test that memory imports work correctly across Python versions."""
    import importlib
    importlib.import_module("crewai.memory.memory")
    importlib.import_module("crewai.memory.external.external_memory")
    
    from crewai.memory.memory import Memory
    
    assert hasattr(Memory, "set_crew")
