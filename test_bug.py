import sys
# Forcing Python to read the precise source directories for both components
sys.path.insert(0, r"C:\Users\zero\crewAI\lib\crewai\src")
sys.path.insert(0, r"C:\Users\zero\crewAI\lib\crewai-core\src")

from crewai import Agent, Crew, Task, Process
from crewai.llms.base_llm import BaseLLM

class StubLLM(BaseLLM):
    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        return "Thought: I know it.\nFinal Answer: The sky is blue."
        
    def supports_function_calling(self) -> bool:
        return False

agent = Agent(role="Tester", goal="Be a Minimal Reproduction", backstory="bg",
              llm=StubLLM(model="stub"), verbose=False)
              
task = Task(description="Sky colour?", expected_output="a colour",
            agent=agent, human_input=True)
            
Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()