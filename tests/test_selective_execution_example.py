"""Example demonstrating selective execution for issue #2941."""

import pytest
from crewai import Agent, Crew, Task, Process


@pytest.mark.vcr(filter_headers=["authorization"])
def test_issue_2941_example():
    """Reproduce and test the exact scenario from issue #2941."""
    
    holiday_agent = Agent(role="Holiday Researcher", goal="Research holidays", backstory="Expert in holidays")
    macro_agent = Agent(role="Macro Analyst", goal="Analyze macro data", backstory="Expert in macroeconomics") 
    news_agent = Agent(role="News Summarizer", goal="Summarize news", backstory="Expert in news analysis")
    forecast_agent = Agent(role="Forecaster", goal="Create forecasts", backstory="Expert in forecasting")
    query_agent = Agent(role="Query Handler", goal="Handle user queries", backstory="Expert in query processing")
    
    holiday_task = Task(description="Research holiday information", expected_output="Holiday data", agent=holiday_agent, tags=["holiday"])
    macro_task = Task(description="Extract macroeconomic data", expected_output="Macro data", agent=macro_agent, tags=["macro"])  
    news_task = Task(description="Summarize relevant news", expected_output="News summary", agent=news_agent, tags=["news"])
    forecast_task = Task(description="Generate forecast", expected_output="Forecast result", agent=forecast_agent, tags=["forecast"])
    query_task = Task(description="Handle user query", expected_output="Query response", agent=query_agent, tags=["query"])
    
    crew = Crew(
        agents=[holiday_agent, macro_agent, news_agent, forecast_agent, query_agent],
        tasks=[holiday_task, macro_task, news_task, forecast_task, query_task],
        process=Process.selective,
        task_selector=Crew.create_tag_selector()
    )
    
    inputs = {
        'data_file': 'sample.csv',
        'action': 'forecast', 
        'country_code': 'US',
        'topic': 'Egg_prices',
        'query': "Provide forecasted result on the input data"
    }
    
    result = crew.kickoff(inputs=inputs)
    assert result is not None


def test_multiple_actions_example():
    """Test crew that can handle multiple different actions."""
    
    researcher = Agent(role="Researcher", goal="Research topics", backstory="Expert researcher")
    analyst = Agent(role="Analyst", goal="Analyze data", backstory="Expert analyst")
    writer = Agent(role="Writer", goal="Write reports", backstory="Expert writer")
    
    research_task = Task(description="Research the topic", expected_output="Research findings", agent=researcher, tags=["research", "data_gathering"])
    analysis_task = Task(description="Analyze the data", expected_output="Analysis results", agent=analyst, tags=["analysis", "forecast"])
    writing_task = Task(description="Write the report", expected_output="Final report", agent=writer, tags=["writing", "summary"])
    
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        task_selector=Crew.create_tag_selector()
    )
    
    research_result = crew.kickoff(inputs={"action": "research"})
    assert research_result is not None
    
    analysis_result = crew.kickoff(inputs={"action": "analysis"})
    assert analysis_result is not None
    
    writing_result = crew.kickoff(inputs={"action": "writing"})
    assert writing_result is not None
