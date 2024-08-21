<div align="center">
# ****

🤖 ****: Cutting-edge framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, AI empowers agents to work together seamlessly, tackling complex tasks.
<h3>
[Homepage](https://www.ai-hive.net/syzygi/) 

</h3>

</div>
## Table of contents

- [License](#license)

## Transform AI Agents

## Introduction
The next generation AI foundation models will achieve reasoning and logic abilities equivalent to PhD level. And while AI doctors, AI lawyers, and AI engineers are not ready to hang out their shingles, every doctor, lawyer, and engineer will want a specialized AI partner to assist them in delivering premium service to their clients. 
​
## The Problem
AI agent teams partnering with professionals face poor coordination, limited adaptability, and inconsistent performance. Trust issues and integration hurdles hinder adoption. AI needs better collaboration mechanisms, adaptive learning, and robust feedback loops to improve. Enhancing communication skills and ethical decision-making is crucial. The goal is to create transparent, flexible AI agent teams that learn continuously, providing reliable assistance across various professional fields.
 
## The Solution
We are developing an AI Agent Team Architecture called Syzygi (pronounced SIZ-in-jee) that mimics some features of the neural net Transformer Architecture used to train LLMs. Syzygi architecture provides power and flexibility for AI agents to synchronize their tasks on one project and train as a team over many projects. As they perform more varied tasks, they become more versatile and efficient as an organization - they learn to become a better team.

## Getting Started

To get started with Syzygi AI, follow these simple steps:

### 1. Installation

```shell
pip install crewai
```

```shell
pip install 'crewai[tools]'
```

```shell
pip install -r requirements.txt
```

```
### 2. Setting Up Your Syzygi Crew based on Crewai

```python
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

def run_crew(user_request):
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["SERPER_API_KEY"] = ""

    search_tool = SerperDevTool()

    researcher = Agent(
        role='Senior Research Analyst',
        goal='Conduct thorough analysis based on the given request',
        backstory="""You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends and analyzing complex topics.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool]
    )
    writer = Agent(
        role='Tech Content Strategist',
        goal='Craft compelling content based on the analysis',
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
        You transform complex concepts into compelling narratives.""",
        verbose=True,
        allow_delegation=True
    )

    task1 = Task(
        description=f"Conduct a comprehensive analysis based on the following request: {user_request}",
        expected_output="Full analysis report in bullet points",
        agent=researcher
    )

    task2 = Task(
        description="""Using the insights provided, develop an engaging blog
        post that highlights the most significant findings from the analysis.
        Your post should be informative yet accessible, catering to a tech-savvy audience.
        Make it sound engaging, avoid complex words so it doesn't sound like AI.""",
        expected_output="Full blog post of at least 4 paragraphs",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=True,
        process=Process.sequential
    )

    result = crew.kickoff()
    return result

def run_postmortem(postmortem_request, previous_result):
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["SERPER_API_KEY"] = ""

    search_tool = SerperDevTool()

    postmortem_analyst = Agent(
        role='Postmortem Analyst',
        goal='Conduct a thorough postmortem analysis of the team\'s performance',
        backstory="""You are an experienced project manager and analyst specializing in team performance and process improvement.
        Your expertise lies in identifying strengths, weaknesses, and areas for improvement in team collaborations.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool]
    )

    postmortem_task = Task(
        description=f"""Analyze the team's performance based on the following request and the previous result:
        Request: {postmortem_request}
        Previous Result: {previous_result}
        
        Provide insights on what went well, what could be improved, and specific recommendations for future tasks.""",
        expected_output="Detailed postmortem analysis with actionable insights",
        agent=postmortem_analyst
    )

    postmortem_crew = Crew(
        agents=[postmortem_analyst],
        tasks=[postmortem_task],
        verbose=True,
        process=Process.sequential
    )

    postmortem_result = postmortem_crew.kickoff()
    return postmortem_result

if __name__ == "__main__":
    # This is just for testing the script directly
    test_request = "Analyze the latest advancements in AI in 2024. Identify key trends, breakthrough technologies, and potential industry impacts."
    result = run_crew(test_request)
    print("######################")
    print(result)
    
    test_postmortem_request = "Conduct a postmortem on the team's performance. How did we do and what could we improve for next time?"
    postmortem_result = run_postmortem(test_postmortem_request, str(result))
    print("######################")
    print(postmortem_result)
```
## 3. run the user interface in streamlit

```shell
streamlit run streamlit_app.py

## Other

In addition to the sequential process, you can use the hierarchical process, which automatically assigns a manager to the defined crew to properly coordinate the planning and execution of tasks through delegation and validation of results. [See more about the processes here](https://docs.crewai.com/core-concepts/Processes/).

## Key Features

- **Role-Based Agent Design**: Customize agents with specific roles, goals, and tools.
- **Autonomous Inter-Agent Delegation**: Agents can autonomously delegate tasks and inquire amongst themselves, enhancing problem-solving efficiency.
- **Flexible Task Management**: Define tasks with customizable tools and assign them to agents dynamically.
- **Processes Driven**: Currently only supports `sequential` task execution and `hierarchical` processes, but more complex processes like consensual and autonomous are being worked on.
- **Save output as file**: Save the output of individual tasks as a file, so you can use it later.
- **Parse output as Pydantic or Json**: Parse the output of individual tasks as a Pydantic model or as a Json if you want to.
- **Works with Open Source Models**: Run your crew using Open AI or open source models refer to the [Connect crewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring your agents' connections to models, even ones running locally!



## Examples



### Quick Tutorial

### Write Job Descriptions

## Connecting Your Crew to a Model
It supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.



## How Syzygi Compares

- **Autogen**: While Autogen does good in creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.

- **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

**CrewAI's Advantage**: CrewAI is built with production in mind. It offers the flexibility of Autogen's conversational agents and the structured process approach of ChatDev, but without the rigidity. CrewAI's processes are designed to be dynamic and adaptable, fitting seamlessly into both development and production workflows.


## Contribution

Syzygi is open-source and we welcome contributions. If you're looking to contribute, please: info@ai-hive.net


- Fork the repository.
- Create a new branch for your feature.
- Add your feature or improvement.
- Send a pull request.
- We appreciate your input!


## License

Syzygi is released under the MIT License.
