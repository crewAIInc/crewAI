from crewai import Agent, Crew, Task, Process, LLM

researcher = Agent(
    role="Researcher",
    goal="Research and analyze information",
    backstory="You are an expert researcher with years of experience.",
    llm=LLM(model="gpt-4o-mini")
)

writer = Agent(
    role="Writer", 
    goal="Write compelling content",
    backstory="You are a skilled writer who creates engaging content.",
    llm=LLM(model="gpt-4o-mini")
)

research_task = Task(
    description="Research the latest trends in AI",
    expected_output="A comprehensive report on AI trends",
    agent=researcher
)

writing_task = Task(
    description="Write an article based on the research",
    expected_output="A well-written article about AI trends",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    trace_execution=True
)

result = crew.kickoff(inputs={"topic": "artificial intelligence"})

if result.execution_trace:
    print(f"Total execution steps: {result.execution_trace.total_steps}")
    print(f"Execution duration: {result.execution_trace.end_time - result.execution_trace.start_time}")
    
    thoughts = result.execution_trace.get_steps_by_type("agent_thought")
    print(f"Agent thoughts captured: {len(thoughts)}")
    
    tool_calls = result.execution_trace.get_steps_by_type("tool_call_started")
    print(f"Tool calls made: {len(tool_calls)}")
    
    for step in result.execution_trace.steps:
        print(f"{step.timestamp}: {step.step_type} - {step.agent_role or 'System'}")
