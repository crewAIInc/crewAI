# 🚀 crewAI: Build & Orchestrate Autonomous AI Agent Crews

<p align="center"><img src="./docs/images/crewai_logo.png" alt="crewAI Logo" width="500"></p>

## Short Description
`crewAI` is a powerful, cutting-edge framework designed to empower developers and researchers to build, run, and orchestrate intelligent, collaborative AI agents. It enables the creation of multi-agent systems that work together seamlessly, tackling complex tasks through collective intelligence, role-playing, and dynamic tool utilization. From sophisticated data analysis to creative content generation, `crewAI` unlocks new dimensions of automation and problem-solving, allowing you to deploy truly autonomous AI teams.

## ✨ Key Features
*   **Intelligent Agent Orchestration:** Define and manage autonomous AI agents with distinct roles, goals, and backstories for specialized expertise and collaborative execution.
*   **Flexible Process Management:** Implement sequential, hierarchical, or custom processes to dictate how agents collaborate, ensuring optimal workflow for any task.
*   **Advanced Tool Integration:** Equip agents with a wide array of tools – from AI/ML functionalities like DALL-E and Code Interpreters to automation tools like Zapier, database access, file manipulation, and web scraping capabilities.
*   **Comprehensive Memory Systems:** Utilize contextual, entity, short-term, long-term, and external memory to ensure agents learn, adapt, and maintain continuity across tasks.
*   **Knowledge Base Integration (RAG):** Seamlessly connect agents to various knowledge sources (PDFs, CSVs, JSON, web pages, vector DBs like ChromaDB and Qdrant) for enhanced retrieval-augmented generation.
*   **LLM Agnostic:** Work with any LLM, supporting a wide range of models including OpenAI, Anthropic, Google Gemini, Ollama, and custom integrations.
*   **Human-in-the-Loop:** Integrate human feedback and intervention points to guide agent workflows and ensure ethical, accurate, and aligned outcomes.
*   **Built-in Observability & Tracing:** Monitor agent interactions, tool usage, and decision-making processes with integrations for tools like Langfuse, MLflow, Arize Phoenix, and more, for complete transparency and debugging.
*   **Enterprise-Grade Capabilities:** Features like hallucination guardrails, agent and tool repositories, RBAC, and webhook streaming make `crewAI` ready for production environments.

## Who is this for?
`crewAI` is ideal for:
*   **AI Developers & Engineers:** Rapidly prototype, develop, and deploy complex multi-agent AI solutions.
*   **AI Researchers:** Experiment with new paradigms in agent collaboration, emergent behavior, and collective intelligence.
*   **Automation Specialists:** Automate intricate business processes that require dynamic decision-making and diverse tool utilization.
*   **Product Teams:** Integrate intelligent AI crews into applications for enhanced user experience, content generation, and data-driven insights.

## Technology Stack & Architecture
`crewAI` is predominantly built with **Python**, leveraging its robust ecosystem for AI and machine learning. At its core, it's designed as an extensible framework with:
*   **Core Components:** Agents, Tasks, Crews, Tools, LLMs, Memory, and Knowledge Bases.
*   **Modularity:** A layered architecture allows for easy customization and extension of individual components.
*   **CLI:** A command-line interface facilitates project creation, deployment, and management.
*   **Data Handling:** Utilizes Pydantic for data validation and parsing, and supports various data storage solutions for RAG and memory.
*   **Workflow Engine:** Orchestrates complex `flows` and `processes` (sequential, hierarchical) using advanced reasoning and planning mechanisms.
*   **Observability:** Integrates with popular tracing and logging platforms for real-time monitoring and debugging.

## 📊 Architecture & Database Schema
The `crewAI` framework is designed around a collaborative multi-agent paradigm, where specialized agents work together, leveraging shared knowledge and tools, orchestrated by a central crew.

```mermaid
graph TD
    User[User Input] --> Crew[Crew (Orchestration Engine)]

    subgraph Crew
        Agent_A[Agent 1: Role, Goal]
        Agent_B[Agent 2: Role, Goal]
        Agent_C[Agent 3: Role, Goal]
        
        Crew -- Delegates Tasks --> Agent_A
        Crew -- Delegates Tasks --> Agent_B
        Crew -- Delegates Tasks --> Agent_C
        
        Agent_A -- Collaborates --> Agent_B
        Agent_B -- Collaborates --> Agent_C
        
        Agent_A -- Performs --> Task_1(Task 1);
        Agent_B -- Performs --> Task_2(Task 2);
        Agent_C -- Performs --> Task_3(Task 3);

        Task_1 -- Requires --> Tool_Set_1(Tools: Search, File I/O);
        Task_2 -- Requires --> Tool_Set_2(Tools: Calculator, API);
        Task_3 -- Requires --> Tool_Set_3(Tools: Vision, Automation);

        Agent_A -- Uses --> LLM_A(LLM A);
        Agent_B -- Uses --> LLM_B(LLM B);
        Agent_C -- Uses --> LLM_C(LLM C);

        Agent_A -- Accesses --> Memory(Memory: Short/Long Term, External);
        Agent_B -- Accesses --> Knowledge(Knowledge Base: RAG);
        Agent_C -- Accesses --> Human(Human in the Loop);
    end

    Crew -- Delivers --> Output[Final Output]
    
    style Crew fill:#f9f,stroke:#333,stroke-width:2px;
    style Agent_A fill:#ccf,stroke:#333,stroke-width:1px;
    style Agent_B fill:#ccf,stroke:#333,stroke-width:1px;
    style Agent_C fill:#ccf,stroke:#333,stroke-width:1px;
    style Task_1 fill:#bfb,stroke:#333,stroke-width:1px;
    style Task_2 fill:#bfb,stroke:#333,stroke-width:1px;
    style Task_3 fill:#bfb,stroke:#333,stroke-width:1px;
    style Tool_Set_1 fill:#fcf,stroke:#333,stroke-width:1px;
    style Tool_Set_2 fill:#fcf,stroke:#333,stroke-width:1px;
    style Tool_Set_3 fill:#fcf,stroke:#333,stroke-width:1px;
    style LLM_A fill:#ffc,stroke:#333,stroke-width:1px;
    style LLM_B fill:#ffc,stroke:#333,stroke-width:1px;
    style LLM_C fill:#ffc,stroke:#333,stroke-width:1px;
    style Memory fill:#fdd,stroke:#333,stroke-width:1px;
    style Knowledge fill:#fdd,stroke:#333,stroke-width:1px;
    style Human fill:#fdd,stroke:#333,stroke-width:1px;
```
This diagram illustrates the core components: User input initiates a `Crew`, which orchestrates multiple `Agents`. Each `Agent` has a defined `Role` and `Goal`, potentially leveraging a specific `LLM` and a set of `Tools` to execute `Tasks`. Agents can `Collaborate` by delegating tasks or sharing information. `Memory` and `Knowledge Bases` (including RAG) provide context and data, while `Human in the Loop` ensures oversight. The collective effort culminates in a `Final Output` for the user.

## ⚡ Quick Start Guide
To get your first AI Crew up and running with `crewAI`, follow these simple steps:

1.  **Installation:**
    Ensure you have Python 3.9 or higher installed.
    ```bash
    pip install crewai
    ```

2.  **Set Up Environment Variables:**
    Set your OpenAI API key (or equivalent for your chosen LLM).
    ```bash
    export OPENAI_API_KEY='YOUR_API_KEY'
    ```

3.  **Create Your First Crew:**
    A simple example defining a `Researcher` agent and a `Writer` agent, collaborating on a research task:

    ```python
    from crewai import Agent, Task, Crew, Process

    # Define your agents
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Uncover groundbreaking insights from various data sources',
        backstory='A meticulous analyst, skilled in extracting critical information and trends.',
        verbose=True,
        allow_delegation=False
    )

    writer = Agent(
        role='Content Strategist',
        goal='Craft compelling and insightful articles',
        backstory='A visionary writer, transforming complex research into engaging narratives.',
        verbose=True,
        allow_delegation=False
    )

    # Define your tasks
    task1 = Task(
        description='Conduct a comprehensive analysis of the latest AI trends in 2024.',
        expected_output='A detailed report on AI advancements and market impact.',
        agent=researcher
    )

    task2 = Task(
        description='Write a blog post about the findings from the AI trends report.',
        expected_output='A 500-word engaging blog post, ready for publication.',
        agent=writer,
        context=[task1]
    )

    # Instantiate your crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=2,
    )

    # Kickoff the crew
    print("CrewAI starting your research and writing process...\n")
    result = crew.kickoff()
    print("\n\n########################")
    print("## Here is the Final Result")
    print("########################\n")
    print(result)
    ```

This example creates two agents, assigns them distinct tasks with dependencies, and runs them as a sequential process.

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.