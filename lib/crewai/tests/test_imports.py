"""Test that all public API classes are properly importable."""


def test_task_output_import():
    """Test that TaskOutput can be imported from crewai."""
    from crewai import TaskOutput

    assert TaskOutput is not None


def test_crew_output_import():
    """Test that CrewOutput can be imported from crewai."""
    from crewai import CrewOutput

    assert CrewOutput is not None


def test_persist_is_importable_from_every_flow_path():
    """AGENTS.md scaffolded by 1.10.0-1.15.28 taught `crewai.flow.flow`."""
    from crewai.flow import persist as from_flow
    from crewai.flow.flow import persist as from_flow_flow
    from crewai.flow.persistence import persist as from_persistence

    assert from_flow_flow is from_flow is from_persistence


def test_persist_is_in_flow_flow_star_exports():
    from crewai.flow import flow

    assert "persist" in flow.__all__
