# 🚀 crewAI

<p align="center"><img src="./docs/images/crewai_logo.png" alt="crewAI Logo" width="500"></p>

## Short Description
crewAI is a revolutionary framework for orchestrating intelligent AI agents, enabling them to collaboratively solve complex tasks through defined roles, shared goals, and streamlined workflows. It empowers developers to build multi-agent systems that go beyond simple task execution, fostering true collaborative intelligence.

## ✨ Key Features
*   **Multi-Agent Orchestration:** Define diverse AI agents with specialized roles, explicit goals, and rich backstories to form powerful, collaborative teams.
*   **Flexible Task Management:** Assign complex tasks that can be executed sequentially or hierarchically, with support for conditional tasks and dynamic flows.
*   **Extensive Tool Integration:** Equip agents with a wide array of predefined and custom tools for interacting with the external world, covering areas like web searching, data processing, automation, and more.
*   **Diverse LLM Support:** Seamlessly integrate with various Large Language Models, including OpenAI, Gemini, Ollama, and custom solutions, providing flexibility in AI capabilities.
*   **Advanced Memory Systems:** Implement sophisticated memory mechanisms including short-term, long-term, contextual, and entity memory, enabling agents to learn, adapt, and make informed decisions.
*   **Retrieval Augmented Generation (RAG):** Enhance agent intelligence by integrating with vector databases like ChromaDB and Qdrant, providing access to up-to-date and highly relevant knowledge.
*   **Pluggable Processes:** Design intricate workflows (sequential, hierarchical) for sophisticated problem-solving, ensuring optimal collaboration and outcome quality.
*   **Observability & Evaluation:** Gain deep insights into agent behavior and crew performance with comprehensive tracing, logging, and evaluation metrics, supporting continuous improvement.
*   **Human-in-the-Loop Capabilities:** Incorporate human oversight and intervention at critical junctures, ensuring alignment with business needs and ethical guidelines.
*   **CLI & Enterprise Features:** Utilize a robust command-line interface for rapid project scaffolding and leverage enterprise-grade functionalities such as Role-Based Access Control (RBAC), agent/tool repositories, webhook streaming, and a wide range of integrations (e.g., Asana, GitHub, HubSpot, Jira, Salesforce, Slack).

## Who is this for?
crewAI is built for software developers, AI engineers, data scientists, and researchers who are looking to:
*   Automate complex, multi-step processes that require reasoning and external interaction.
*   Develop intelligent applications that go beyond single-prompt interactions.
*   Build AI systems that can adapt and learn over time.
*   Experiment with advanced agentic architectures and collaborative AI.
*   Integrate AI into existing enterprise workflows and tools.

## Technology Stack & Architecture
crewAI is primarily built in Python, leveraging its rich ecosystem for AI and machine learning. Its modular design allows for high flexibility and extensibility.

**Core Components:**
*   **Python:** The foundational language for the entire framework.
*   **Large Language Models (LLMs):** Integrates with major LLM providers (OpenAI, Google Gemini) and local models (Ollama), with an extensible interface for custom LLMs.
*   **Pydantic:** Used extensively for data validation and settings management.
*   **LangChain:** Portions of the framework are inspired by or integrate with LangChain concepts for agent tooling.

**Architectural Overview:**
The core architecture of crewAI revolves around the orchestration of multiple AI agents. Agents are given roles, goals, and backstories, and they collaborate to complete tasks within a defined process.

## 📊 Architecture & Database Schema
```mermaid
graph TD
    A[User Inputs/Triggers] --> B(Flow Orchestrator)
    B --> C{Crew Definition}
    C --> D[Agents]
    C --> E[Tasks]
    D -- "Execute" --> E
    D -- "Utilizes" --> F[Tools]
    D -- "Consults" --> G[Memory Systems]
    D -- "Accesses" --> H[Knowledge Base (RAG)]
    D -- "Powered by" --> I[Large Language Models]
    E -- "Generates Output" --> J[Final Result]
    J --> K(Observability & Evaluation)

    %% Memory Sub-types
    G -- "Short-term" --> G1
    G -- "Long-term" --> G2
    G -- "Contextual" --> G3
    G -- "Entity" --> G4

    %% Knowledge Base Integrations
    H -- "ChromaDB" --> H1
    H -- "Qdrant" --> H2

    %% Tool Categories
    F -- "AI/ML" --> F1
    F -- "Automation" --> F2
    F -- "Databases" --> F3
    F -- "Web Scraping" --> F4

    %% LLM Providers
    I -- "OpenAI" --> I1
    I -- "Google Gemini" --> I2
    I -- "Ollama" --> I3
    I -- "Custom LLMs" --> I4
```

This flowchart illustrates the core components and their interactions: user inputs or external triggers initiate a flow, which orchestrates a crew of specialized agents and tasks. Agents perform tasks by utilizing various tools, consulting different memory systems, accessing knowledge bases via RAG, and leveraging Large Language Models for reasoning and generation. The entire process is monitored and improved through observability and evaluation mechanisms, leading to a final result.

## ⚡ Quick Start Guide
Get your first crew up and running in no time!

1.  **Installation**
    First, install `crewai` using pip:
    ```bash
    pip install crewai
    ```
    If you plan to use specific LLMs or tools, you might need additional installations (e.g., `pip install crewai[openai]` or `pip install crewai-tools`).

2.  **Environment Setup**
    Set your LLM API keys as environment variables. For instance, if using OpenAI:
    ```bash
    export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
    # Or for other models, e.g., for Google Gemini:
    export GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
    ```

3.  **Define Your Crew (Python Example)**
    Create a Python file (e.g., `my_crew.py`) and define your agents, tasks, and the crew. This example uses `crewai-tools` for searching.
    ```python
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import SerperDevTool # Example tool

    # Initialize tools if needed
    search_tool = SerperDevTool()

    # 1. Define your Agents
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Uncover groundbreaking technologies in AI and automation',
        backstory='A seasoned analyst dedicated to identifying emerging tech trends.',
        verbose=True,
        allow_delegation=False, # This agent does not delegate tasks
        tools=[search_tool] # Assign the search tool
    )

    writer = Agent(
        role='Tech Content Strategist',
        goal='Craft compelling narratives about innovative tech advancements',
        backstory='An eloquent writer who transforms complex technical concepts into engaging stories.',
        verbose=True,
        allow_delegation=True # This agent can delegate if needed
    )

    # 2. Define your Tasks
    research_task = Task(
        description='Identify the top 5 most promising AI technologies disrupting the market in 2024.',
        expected_output='A detailed report on the top 5 AI technologies, their potential impact, and key players.',
        agent=researcher
    )

    write_report_task = Task(
        description='Write a compelling blog post (1000 words) based on the research findings. Focus on explaining the technologies simply.',
        expected_output='A 1000-word blog post, optimized for readability and clarity, summarizing the research.',
        agent=writer
    )

    # 3. Instantiate your Crew
    tech_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_report_task],
        process=Process.sequential, # Tasks will be executed one after another
        verbose=2, # Detailed logging
        full_output=True # Get full output details
    )

    # 4. Kickoff the Crew
    print("Starting the Tech Crew to identify AI breakthroughs...")
    result = tech_crew.kickoff(inputs={'topic': 'AI in industrial automation'})
    print("\n\n########################")
    print("## Final Crew Output:")
    print("########################\n")
    print(result)
    ```

4.  **Run Your Crew**
    Execute your Python script:
    ```bash
    python my_crew.py
    ```
    Watch as your agents collaborate to complete the tasks and generate the final output!

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.