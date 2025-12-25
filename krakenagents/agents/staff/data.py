"""STAFF-08: Head of Data & Intelligence Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools import (
    get_spot_research_tools,
    get_futures_research_tools,
)


def create_data_agent() -> Agent:
    """Create STAFF-08 Head of Data & Intelligence Agent.

    Responsible for:
    - Data pipelines and ingestion
    - Dashboards for all desks
    - Data quality and anomaly detection
    - Alternative data and research datasets

    Reports to: STAFF-00 (CEO)
    Uses light LLM for data operations.
    """
    # Research/market data tools for data analysis
    tools = get_spot_research_tools() + get_futures_research_tools()

    return create_light_agent(
        role="Head of Data & Intelligence — Data Pipelines, Dashboards, Alt-Data",
        goal=(
            "Manage data pipelines, dashboards, alternative data, and QA for all desks. "
            "Set up core datasets with QA checks. Standardize desk dashboards (risk/perf/"
            "execution/research). Alert on data pollution and outliers. Introduce alternative "
            "data (social sentiment, search trends, developer metrics) and integrate into "
            "dashboards. Experiment with machine learning (predictive models, anomaly detection) "
            "to find hidden alpha and validate signals for use."
        ),
        backstory=(
            "Data engineering and analytics leader with expertise in building trading data "
            "infrastructure. Strong background in real-time data pipelines, data quality "
            "frameworks, and visualization. Experience with crypto-specific data sources "
            "(on-chain data, DEX data, social sentiment). Known for building reliable data "
            "systems that traders trust. Interested in machine learning applications for "
            "alpha generation."
        ),
        tools=tools,
        allow_delegation=False,
    )
