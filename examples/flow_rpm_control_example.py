#!/usr/bin/env python3
"""
Example demonstrating Flow-level RPM control in CrewAI.

This example shows how to use the new max_rpm parameter at the Flow level
to control the global request rate across all Crews within a Flow.
"""

import logging
from typing import Dict, Any, Optional

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start, listen

# Configuration constants
FLOW_GLOBAL_RPM = 10
UNLIMITED_CREW_RPM = 50
MANUAL_CREW_RPM = 25

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalysisFlow(Flow):
    """Example flow with global RPM control."""

    def __init__(self) -> None:
        # Initialize Flow with global RPM limit of 10 requests per minute
        # This limit will be applied across ALL crews in this flow
        super().__init__(max_rpm=FLOW_GLOBAL_RPM, verbose=True)

    @start()
    def initialize_analysis(self) -> Dict[str, str]:
        """Initialize the analysis process."""
        logger.info(f"🚀 Starting analysis flow with global RPM limit of {FLOW_GLOBAL_RPM} RPM")
        return {"status": "initialized", "data_source": "production_db"}

    @listen(initialize_analysis)
    def run_data_collection_crew(self, context: Dict[str, str]) -> Any:
        """Run the data collection crew."""
        logger.info("📊 Running data collection crew")

        # Create agents for data collection
        data_analyst = Agent(
            role="Data Analyst",
            goal="Collect and validate data from various sources",
            backstory="Expert in data extraction and validation",
            max_rpm=20  # This will be overridden by Flow's global limit of 10
        )

        data_validator = Agent(
            role="Data Validator",
            goal="Ensure data quality and completeness",
            backstory="Specialist in data quality assurance"
        )

        # Create tasks
        collect_task = Task(
            description="Collect data from the production database",
            agent=data_analyst,
            expected_output="Clean dataset with validation report"
        )

        validate_task = Task(
            description="Validate data quality and completeness",
            agent=data_validator,
            expected_output="Data quality report with metrics"
        )

        # Create crew - this will automatically use Flow's global RPM limit
        data_crew = Crew(
            agents=[data_analyst, data_validator],
            tasks=[collect_task, validate_task],
            max_rpm=15  # This will be overridden by Flow's global limit
        )

        # Execute the crew
        result = data_crew.kickoff()
        return result

    @listen(run_data_collection_crew)
    def run_analysis_crew(self, context: Any) -> Any:
        """Run the analysis crew."""
        logger.info("🔍 Running analysis crew")

        # Create agents for analysis
        senior_analyst = Agent(
            role="Senior Data Analyst",
            goal="Perform comprehensive data analysis",
            backstory="Senior analyst with expertise in statistical analysis"
        )

        insights_specialist = Agent(
            role="Insights Specialist",
            goal="Extract actionable insights from analysis",
            backstory="Expert in translating data into business insights"
        )

        # Create tasks
        analyze_task = Task(
            description="Perform statistical analysis on the validated data",
            agent=senior_analyst,
            expected_output="Statistical analysis report with findings"
        )

        insights_task = Task(
            description="Extract key insights and recommendations",
            agent=insights_specialist,
            expected_output="Business insights report with recommendations"
        )

        # Create crew - this will also use Flow's global RPM limit
        analysis_crew = Crew(
            agents=[senior_analyst, insights_specialist],
            tasks=[analyze_task, insights_task],
            max_rpm=8  # This will be overridden by Flow's global limit
        )

        # Execute the crew
        result = analysis_crew.kickoff()
        return result

    @listen(run_analysis_crew)
    def finalize_report(self, context: Any) -> Dict[str, str]:
        """Finalize the analysis report."""
        logger.info("📋 Finalizing analysis report")
        return {"status": "completed", "report": "final_analysis_report.pdf"}


# Example of Flow without RPM control for comparison
class UnlimitedFlow(Flow):
    """Example flow without RPM control."""

    @start()
    def process_data(self) -> Any:
        """Process data without RPM limits."""
        logger.info("🏃‍♂️ Running unlimited flow")

        agent = Agent(
            role="Data Processor",
            goal="Process large volumes of data quickly",
            backstory="High-performance data processing specialist"
        )

        task = Task(
            description="Process large dataset",
            agent=agent,
            expected_output="Processed data"
        )

        # This crew will use its own RPM settings (or no limits)
        crew = Crew(
            agents=[agent],
            tasks=[task],
            max_rpm=UNLIMITED_CREW_RPM  # This will be respected since Flow has no global limit
        )

        return crew.kickoff()


def demonstrate_flow_rpm_control() -> None:
    """Demonstrate Flow-level RPM control."""
    logger.info("🔒 Example 1: Flow with Global RPM Control (%d RPM)", FLOW_GLOBAL_RPM)
    logger.info("-" * 50)

    analysis_flow = AnalysisFlow()

    # Show the Flow's RPM controller
    if analysis_flow.get_flow_rpm_controller():
        logger.info("✅ Flow has global RPM controller: %d RPM", analysis_flow.max_rpm)
    else:
        logger.warning("❌ Flow has no RPM controller")

    # Execute the flow
    try:
        result = analysis_flow.kickoff()
        logger.info("✅ Flow completed successfully: %s", result)
    except Exception as e:
        logger.error("❌ Flow failed: %s", e)
        raise
    finally:
        # Clean up resources
        analysis_flow.cleanup_resources()


def demonstrate_unlimited_flow() -> None:
    """Demonstrate Flow without RPM control."""
    logger.info("🚀 Example 2: Flow without RPM Control")
    logger.info("-" * 50)

    unlimited_flow = UnlimitedFlow()

    # Show the Flow's RPM controller status
    if unlimited_flow.get_flow_rpm_controller():
        logger.info("✅ Flow has global RPM controller: %d RPM", unlimited_flow.max_rpm)
    else:
        logger.info("❌ Flow has no global RPM controller - crews use individual limits")

    # Execute the flow
    try:
        result = unlimited_flow.kickoff()
        logger.info("✅ Flow completed successfully: %s", result)
    except Exception as e:
        logger.error("❌ Flow failed: %s", e)
        raise
    finally:
        # Clean up resources
        unlimited_flow.cleanup_resources()


def demonstrate_manual_configuration() -> None:
    """Demonstrate manual crew configuration."""
    logger.info("🔧 Example 3: Manual Crew Configuration")
    logger.info("-" * 50)

    # Create a flow with RPM control
    flow_with_rpm = AnalysisFlow()

    try:
        # Create a crew manually
        manual_agent = Agent(
            role="Manual Agent",
            goal="Test manual configuration",
            backstory="Agent for testing manual RPM configuration"
        )

        manual_task = Task(
            description="Test manual crew configuration",
            agent=manual_agent,
            expected_output="Configuration test result"
        )

        manual_crew = Crew(
            agents=[manual_agent],
            tasks=[manual_task],
            max_rpm=MANUAL_CREW_RPM  # This will be overridden
        )

        logger.info("Before configuration: Crew RPM = %d", manual_crew.max_rpm)

        # Manually configure the crew with flow's RPM controller
        flow_with_rpm.set_crew_rpm_controller(manual_crew)

        logger.info("After configuration: Crew now uses Flow's global RPM limit")
        logger.info("Flow RPM limit: %d", flow_with_rpm.max_rpm)

    finally:
        # Clean up resources
        flow_with_rpm.cleanup_resources()


def main() -> None:
    """Run the example flows."""
    try:
        logger.info("=" * 60)
        logger.info("CrewAI Flow-Level RPM Control Example")
        logger.info("=" * 60)

        # Example 1: Flow with global RPM control
        demonstrate_flow_rpm_control()

        logger.info("\n" + "=" * 60)

        # Example 2: Flow without RPM control
        demonstrate_unlimited_flow()

        logger.info("\n" + "=" * 60)

        # Example 3: Manual crew configuration
        demonstrate_manual_configuration()

        logger.info("\n" + "=" * 60)
        logger.info("Example completed!")

    except Exception as e:
        logger.error("Unexpected error occurred: %s", e)
        raise


if __name__ == "__main__":
    main()
