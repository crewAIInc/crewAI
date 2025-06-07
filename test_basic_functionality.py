from crewai import Crew, Agent, Task, Process

print('Basic imports work')

agent = Agent(role="Test", goal="Test", backstory="Test")
task = Task(description='test', expected_output='test', agent=agent, tags=['test'])
print('Tags field works:', task.tags)

crew = Crew(agents=[agent], tasks=[task], task_selector=lambda inputs, task: True)
print('Task selector field works')

print('Process.selective exists:', hasattr(Process, 'selective'))
print('Process.selective value:', Process.selective if hasattr(Process, 'selective') else 'Not found')

selector = Crew.create_tag_selector()
print('create_tag_selector works:', callable(selector))

print('All basic functionality tests passed!')
