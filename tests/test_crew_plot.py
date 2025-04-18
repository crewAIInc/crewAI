import pytest

from crewai.crew import Crew
from crewai.agent import Agent


def test_crew_plot_method_not_implemented():
    """Test that trying to plot a Crew raises the correct error."""
    agent = Agent(
        role="Test Agent",
        goal="Test Goal",
        backstory="Test Backstory"
    )
    crew = Crew(agents=[agent], tasks=[])
    
    with pytest.raises(NotImplementedError) as excinfo:
        crew.plot()
    
    assert "plot method is not available for Crew objects" in str(excinfo.value)
    assert "Plot functionality is only available for Flow objects" in str(excinfo.value)
