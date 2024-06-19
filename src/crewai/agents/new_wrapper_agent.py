from crewai import LangchainAgent, Task, Crew

agent1 = LangchainAgent(
    role="backstory agent",
    goal="who is {input}?",
    backstory="agent backstory",
    verbose=True,
)

task1 = Task(
    expected_output="a short biography of {input}",
    description="a short biography of {input}",
    agent=agent1,
)

agent2 = LangchainAgent(
    role="bio agent",
    goal="summarize the short bio for {input}",
    backstory="agent backstory",
    verbose=True,
)

task2 = Task(
    description="a tldr summary of the short biography",
    expected_output="5 bullet point summary of the biography",
    agent=agent2,
    context=[task1],
)

# taskcomp1 = agent1.execute_task(task1)
# taskcomp2 = agent2.execute_task(task2)
# print('taskcomp1',taskcomp1)
# print('taskcomp2',taskcomp2)


my_crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
crew = my_crew.kickoff(inputs={"input": "andrew ng"})
print("crew", crew)
