"""
Example demonstrating the formal responsibility tracking system in CrewAI.

This example shows how to:
1. Set up agents with capabilities
2. Use responsibility-based task assignment
3. Monitor accountability and performance
4. Generate system insights and recommendations
"""

from crewai import Agent, Crew, Task
from crewai.responsibility.models import AgentCapability, CapabilityType, TaskRequirement
from crewai.responsibility.system import ResponsibilitySystem
from crewai.responsibility.assignment import AssignmentStrategy


def create_agents_with_capabilities():
    """Create agents with defined capabilities."""
    
    python_capabilities = [
        AgentCapability(
            name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.9,
            confidence_score=0.8,
            description="Expert in Python development and scripting",
            keywords=["python", "programming", "development", "scripting"]
        ),
        AgentCapability(
            name="Web Development",
            capability_type=CapabilityType.TECHNICAL,
            proficiency_level=0.7,
            confidence_score=0.7,
            description="Experience with web frameworks",
            keywords=["web", "flask", "django", "fastapi"]
        )
    ]
    
    python_agent = Agent(
        role="Python Developer",
        goal="Develop high-quality Python applications and scripts",
        backstory="Experienced Python developer with expertise in various frameworks",
        capabilities=python_capabilities
    )
    
    analysis_capabilities = [
        AgentCapability(
            name="Data Analysis",
            capability_type=CapabilityType.ANALYTICAL,
            proficiency_level=0.9,
            confidence_score=0.9,
            description="Expert in statistical analysis and data interpretation",
            keywords=["data", "analysis", "statistics", "pandas", "numpy"]
        ),
        AgentCapability(
            name="Machine Learning",
            capability_type=CapabilityType.ANALYTICAL,
            proficiency_level=0.8,
            confidence_score=0.7,
            description="Experience with ML algorithms and model building",
            keywords=["machine learning", "ml", "scikit-learn", "tensorflow"]
        )
    ]
    
    analyst_agent = Agent(
        role="Data Analyst",
        goal="Extract insights from data and build predictive models",
        backstory="Data scientist with strong statistical background",
        capabilities=analysis_capabilities
    )
    
    management_capabilities = [
        AgentCapability(
            name="Project Management",
            capability_type=CapabilityType.LEADERSHIP,
            proficiency_level=0.8,
            confidence_score=0.9,
            description="Experienced in managing technical projects",
            keywords=["project management", "coordination", "planning"]
        ),
        AgentCapability(
            name="Communication",
            capability_type=CapabilityType.COMMUNICATION,
            proficiency_level=0.9,
            confidence_score=0.8,
            description="Excellent communication and coordination skills",
            keywords=["communication", "coordination", "stakeholder management"]
        )
    ]
    
    manager_agent = Agent(
        role="Project Manager",
        goal="Coordinate team efforts and ensure project success",
        backstory="Experienced project manager with technical background",
        capabilities=management_capabilities
    )
    
    return [python_agent, analyst_agent, manager_agent]


def create_tasks_with_requirements():
    """Create tasks with specific capability requirements."""
    
    data_processing_requirements = [
        TaskRequirement(
            capability_name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            minimum_proficiency=0.7,
            weight=1.0,
            keywords=["python", "programming"]
        ),
        TaskRequirement(
            capability_name="Data Analysis",
            capability_type=CapabilityType.ANALYTICAL,
            minimum_proficiency=0.6,
            weight=0.8,
            keywords=["data", "analysis"]
        )
    ]
    
    data_task = Task(
        description="Create a Python script to process and analyze customer data",
        expected_output="A Python script that processes CSV data and generates summary statistics"
    )
    
    web_dashboard_requirements = [
        TaskRequirement(
            capability_name="Web Development",
            capability_type=CapabilityType.TECHNICAL,
            minimum_proficiency=0.6,
            weight=1.0,
            keywords=["web", "development"]
        ),
        TaskRequirement(
            capability_name="Python Programming",
            capability_type=CapabilityType.TECHNICAL,
            minimum_proficiency=0.5,
            weight=0.7,
            keywords=["python", "programming"]
        )
    ]
    
    web_task = Task(
        description="Create a web dashboard to visualize data analysis results",
        expected_output="A web application with interactive charts and data visualization"
    )
    
    coordination_requirements = [
        TaskRequirement(
            capability_name="Project Management",
            capability_type=CapabilityType.LEADERSHIP,
            minimum_proficiency=0.7,
            weight=1.0,
            keywords=["project management", "coordination"]
        ),
        TaskRequirement(
            capability_name="Communication",
            capability_type=CapabilityType.COMMUNICATION,
            minimum_proficiency=0.8,
            weight=0.9,
            keywords=["communication", "coordination"]
        )
    ]
    
    coordination_task = Task(
        description="Coordinate the team efforts and ensure project milestones are met",
        expected_output="Project status report with timeline and deliverable tracking"
    )
    
    return [
        (data_task, data_processing_requirements),
        (web_task, web_dashboard_requirements),
        (coordination_task, coordination_requirements)
    ]


def demonstrate_responsibility_tracking():
    """Demonstrate the complete responsibility tracking workflow."""
    
    print("🚀 CrewAI Formal Responsibility Tracking System Demo")
    print("=" * 60)
    
    print("\n1. Creating agents with defined capabilities...")
    agents = create_agents_with_capabilities()
    
    for agent in agents:
        print(f"   ✓ {agent.role}: {len(agent.capabilities)} capabilities")
    
    print("\n2. Setting up crew with responsibility tracking...")
    crew = Crew(
        agents=agents,
        tasks=[],
        verbose=True
    )
    
    responsibility_system = crew.responsibility_system
    print(f"   ✓ Responsibility system enabled: {responsibility_system.enabled}")
    
    print("\n3. System overview:")
    overview = responsibility_system.get_system_overview()
    print(f"   • Total agents: {overview['total_agents']}")
    print(f"   • Capability distribution: {overview['capability_distribution']}")
    
    print("\n4. Creating tasks with capability requirements...")
    tasks_with_requirements = create_tasks_with_requirements()
    
    print("\n5. Demonstrating responsibility assignment strategies...")
    
    for i, (task, requirements) in enumerate(tasks_with_requirements):
        print(f"\n   Task {i+1}: {task.description[:50]}...")
        
        for strategy in [AssignmentStrategy.GREEDY, AssignmentStrategy.BALANCED, AssignmentStrategy.OPTIMAL]:
            assignment = responsibility_system.assign_task_responsibility(
                task, requirements, strategy
            )
            
            if assignment:
                agent = responsibility_system._get_agent_by_id(assignment.agent_id)
                print(f"   • {strategy.value}: {agent.role} (score: {assignment.responsibility_score:.3f})")
                print(f"     Capabilities matched: {', '.join(assignment.capability_matches)}")
                
                responsibility_system.complete_task(
                    agent=agent,
                    task=task,
                    success=True,
                    completion_time=1800.0,
                    quality_score=0.85,
                    outcome_description="Task completed successfully"
                )
            else:
                print(f"   • {strategy.value}: No suitable agent found")
    
    print("\n6. Agent status and performance:")
    for agent in agents:
        status = responsibility_system.get_agent_status(agent)
        print(f"\n   {agent.role}:")
        print(f"   • Current workload: {status['current_workload']}")
        if status['performance']:
            perf = status['performance']
            print(f"   • Success rate: {perf['success_rate']:.2f}")
            print(f"   • Quality score: {perf['quality_score']:.2f}")
            print(f"   • Total tasks: {perf['total_tasks']}")
    
    print("\n7. Accountability tracking:")
    for agent in agents:
        report = responsibility_system.accountability.generate_accountability_report(agent=agent)
        if report['total_records'] > 0:
            print(f"\n   {agent.role} accountability:")
            print(f"   • Total records: {report['total_records']}")
            print(f"   • Action types: {list(report['action_counts'].keys())}")
            print(f"   • Recent actions: {len(report['recent_actions'])}")
    
    print("\n8. System recommendations:")
    recommendations = responsibility_system.generate_recommendations()
    if recommendations:
        for rec in recommendations:
            print(f"   • {rec['type']}: {rec['description']} (Priority: {rec['priority']})")
    else:
        print("   • No recommendations at this time")
    
    print("\n9. Demonstrating task delegation:")
    if len(agents) >= 2:
        delegation_task = Task(
            description="Complex task requiring delegation",
            expected_output="Delegated task completion report"
        )
        
        responsibility_system.delegate_task(
            delegating_agent=agents[0],
            receiving_agent=agents[1],
            task=delegation_task,
            reason="Specialized expertise required"
        )
        
        print(f"   ✓ Delegated task from {agents[0].role} to {agents[1].role}")
        
        delegation_records = responsibility_system.accountability.get_delegation_chain(delegation_task)
        print(f"   • Delegation chain length: {len(delegation_records)}")
    
    print("\n" + "=" * 60)
    print("🎉 Responsibility tracking demonstration completed!")
    print("\nKey features demonstrated:")
    print("• Capability-based agent hierarchy")
    print("• Mathematical responsibility assignment")
    print("• Accountability logging")
    print("• Performance-based capability adjustment")


if __name__ == "__main__":
    try:
        demonstrate_responsibility_tracking()
        print("\n✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
