import pytest
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

from crewai.experimental.evaluation.testing import (
    assert_experiment_successfully,
    run_experiment,
)


@pytest.fixture
def financial_analysis_crew():
    search_tool = SerperDevTool()

    data_researcher = Agent(
        role="Financial Data Researcher",
        goal="Efficiently collect and structure key financial metrics using multiple search strategies. Using only the search tool.",
        backstory=(
            "You are a precision-focused financial analyst who uses multiple targeted searches "
            "to cross-verify data and ensure comprehensive coverage. You leverage different "
            "search approaches to gather financial information from various authoritative sources."
        ),
        tools=[search_tool],
    )

    financial_analyst = Agent(
        role="Financial Analyst",
        goal="Analyze financial data to assess company performance and outlook",
        backstory=(
            "You are a seasoned financial analyst with expertise in evaluating company "
            "performance through quantitative analysis. You can interpret financial statements, "
            "identify trends, and make reasoned assessments of a company's financial health."
        ),
        tools=[search_tool],
    )

    report_writer = Agent(
        role="Financial Report Writer",
        goal="Synthesize financial analysis into clear, actionable reports",
        backstory=(
            "You are an experienced financial writer who excels at turning complex financial "
            "analyses into clear, concise, and impactful reports. You know how to highlight "
            "key insights and present information in a way that's accessible to various audiences."
        ),
        tools=[],
    )

    research_task = Task(
        description=(
            "Research {company} financial data using multiple targeted search queries:\n\n"
            "**Search Strategy - Execute these searches sequentially:**\n"
            "1. '{company} quarterly earnings Q4 2024 Q1 2025 financial results'\n"
            "2. '{company} financial metrics P/E ratio profit margin debt equity'\n"
            "3. '{company} revenue growth year over year earnings growth rate'\n"
            "4. '{company} recent financial news SEC filings analyst reports'\n"
            "5. '{company} stock performance market cap valuation 2024 2025'\n\n"
            "**Data Collection Guidelines:**\n"
            "- Use multiple search queries to cross-verify financial figures\n"
            "- Prioritize official sources (SEC filings, earnings calls, company reports)\n"
            "- Compare data across different financial platforms for accuracy\n"
            "- Present findings in the exact format specified in expected_output."
        ),
        expected_output=(
            "Financial data summary in this structure:\n\n"
            "## Company Financial Overview\n"
            "**Data Sources Used:** [List 3-5 sources from multiple searches]\n\n"
            "**Latest Quarter:** [Period]\n"
            "- Revenue: $X (YoY: +/-X%) [Source verification]\n"
            "- Net Income: $X (YoY: +/-X%) [Source verification]\n"
            "- EPS: $X (YoY: +/-X%) [Source verification]\n\n"
            "**Key Metrics:**\n"
            "- P/E Ratio: X [Current vs Historical]\n"
            "- Profit Margin: X% [Trend indicator]\n"
            "- Debt-to-Equity: X [Industry comparison]\n\n"
            "**Growth Analysis:**\n"
            "- Revenue Growth: X% (3-year trend)\n"
            "- Earnings Growth: X% (consistency check)\n\n"
            "**Material Developments:** [1-2 key items with impact assessment]\n"
            "**Data Confidence:** [High/Medium/Low based on source consistency]"
        ),
        agent=data_researcher,
    )

    analysis_task = Task(
        description=(
            "Analyze the collected financial data to assess the company's performance and outlook. "
            "Include the following in your analysis:\n"
            "1. Evaluation of financial health based on key metrics\n"
            "2. Trend analysis showing growth or decline patterns\n"
            "3. Comparison with industry benchmarks or competitors\n"
            "4. Identification of strengths and potential areas of concern\n"
            "5. Short-term financial outlook based on current trends"
        ),
        expected_output=(
            "A detailed financial analysis that includes assessment of key metrics, trends, "
            "comparative analysis, and a reasoned outlook for the company's financial future."
        ),
        agent=financial_analyst,
        context=[research_task],
    )

    report_task = Task(
        description=(
            "Create a professional financial report based on the research and analysis. "
            "The report should:\n"
            "1. Begin with an executive summary highlighting key findings\n"
            "2. Present the financial analysis in a clear, logical structure\n"
            "3. Include visual representations of key data points (described textually)\n"
            "4. Provide actionable insights for potential investors\n"
            "5. Conclude with a clear investment recommendation (buy, hold, or sell)"
        ),
        expected_output=(
            "A professional, comprehensive financial report with executive summary, "
            "structured analysis, visual elements, actionable insights, and a clear recommendation."
        ),
        agent=report_writer,
        context=[research_task, analysis_task],
    )

    crew = Crew(
        agents=[data_researcher, financial_analyst, report_writer],
        tasks=[research_task, analysis_task, report_task],
        process=Process.sequential,
    )

    return crew


def test_financial_analysis_regression(financial_analysis_crew):
    dataset = [
        {
            "inputs": {"company": "Apple Inc. (AAPL)"},
            "expected_score": {"goal_alignment": 8},
        },
        {
            "identifier": "test_2",
            "inputs": {"company": "Microsoft Corporation (MSFT)"},
            "expected_score": 8,
        },
    ]

    results = run_experiment(dataset=dataset, crew=financial_analysis_crew, verbose=True)

    assert_experiment_successfully(results)
