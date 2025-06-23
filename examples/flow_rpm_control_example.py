#!/usr/bin/env python3
"""
Example demonstrating Flow-level RPM control in CrewAI.

This example shows how to use the new max_rpm parameter at the Flow level
to control the global request rate across all Crews within a Flow.
"""

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start, listen


class AnalysisFlow(Flow):
    """Example flow with global RPM control."""

    def __init__(self):
        # Initialize Flow with global RPM limit of 10 requests per minute
        # This limit will be applied across ALL crews in this flow
        super().__init__(max_rpm=10, verbose=True)

    @start()
    def initialize_analysis(self):
        """Initialize the analysis process."""
        print("🚀 Starting analysis flow with global RPM limit of 10 RPM")
        return {"status": "initialized", "data_source": "production_db"}

    @listen(initialize_analysis)
    def run_data_collection_crew(self, context):
        """Run the data collection crew."""
        print("📊 Running data collection crew")

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
            max_rpm=15  # This will be overridden by Flow's global limit of 10
        )

        # Execute the crew
        result = data_crew.kickoff()
        return result

    @listen(run_data_collection_crew)
    def run_analysis_crew(self, context):
        """Run the analysis crew."""
        print("🔍 Running analysis crew")

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
            max_rpm=8  # This will be overridden by Flow's global limit of 10
        )

        # Execute the crew
        result = analysis_crew.kickoff()
        return result

    @listen(run_analysis_crew)
    def finalize_report(self, context):
        """Finalize the analysis report."""
        print("📋 Finalizing analysis report")
        return {"status": "completed", "report": "final_analysis_report.pdf"}


# Example of Flow without RPM control for comparison
class UnlimitedFlow(Flow):
    """Example flow without RPM control."""

    @start()
    def process_data(self):
        """Process data without RPM limits."""
        print("🏃‍♂️ Running unlimited flow")

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
            max_rpm=50  # This will be respected since Flow has no global limit
        )

        return crew.kickoff()


def main():
    """Run the example flows."""
    print("=" * 60)
    print("CrewAI Flow-Level RPM Control Example")
    print("=" * 60)

    # Example 1: Flow with global RPM control
    print("\n🔒 Example 1: Flow with Global RPM Control (10 RPM)")
    print("-" * 50)

    analysis_flow = AnalysisFlow()

    # Show the Flow's RPM controller
    if analysis_flow.get_flow_rpm_controller():
        print(f"✅ Flow has global RPM controller: {analysis_flow.max_rpm} RPM")
    else:
        print("❌ Flow has no RPM controller")

    # Execute the flow
    try:
        result = analysis_flow.kickoff()
        print(f"✅ Flow completed successfully: {result}")
    except Exception as e:
        print(f"❌ Flow failed: {e}")

    print("\n" + "=" * 60)

    # Example 2: Flow without RPM control
    print("\n🚀 Example 2: Flow without RPM Control")
    print("-" * 50)

    unlimited_flow = UnlimitedFlow()

    # Show the Flow's RPM controller status
    if unlimited_flow.get_flow_rpm_controller():
        print(f"✅ Flow has global RPM controller: {unlimited_flow.max_rpm} RPM")
    else:
        print("❌ Flow has no global RPM controller - crews use individual limits")

    # Execute the flow
    try:
        result = unlimited_flow.kickoff()
        print(f"✅ Flow completed successfully: {result}")
    except Exception as e:
        print(f"❌ Flow failed: {e}")

    print("\n" + "=" * 60)
    print("Example completed!")

    # Demonstrate manual crew configuration
    print("\n🔧 Example 3: Manual Crew Configuration")
    print("-" * 50)

    # Create a flow with RPM control
    flow_with_rpm = AnalysisFlow()

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
        max_rpm=25  # This will be overridden
    )

    print(f"Before configuration: Crew RPM = {manual_crew.max_rpm}")

    # Manually configure the crew with flow's RPM controller
    flow_with_rpm.set_crew_rpm_controller(manual_crew)

    print(f"After configuration: Crew now uses Flow's global RPM limit")
    print(f"Flow RPM limit: {flow_with_rpm.max_rpm}")


if __name__ == "__main__":
    main()
