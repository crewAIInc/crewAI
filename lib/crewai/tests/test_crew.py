"""Test Agent creation and execution basic functionality."""

from io import StringIO
import json
import threading
from collections import defaultdict
from concurrent.futures import Future
from hashlib import md5
import re
import sys
from unittest.mock import ANY, MagicMock, call, patch

from crewai.agent import Agent
from crewai.agents import CacheHandler
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.crew_events import (
    CrewTestCompletedEvent,
    CrewTestStartedEvent,
    CrewTrainCompletedEvent,
    CrewTrainStartedEvent,
)
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryFailedEvent,
    MemoryQueryStartedEvent,
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveFailedEvent,
    MemorySaveStartedEvent,
)
from crewai.flow import Flow, start
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.llm import LLM
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.memory.external.external_memory import ExternalMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.process import Process
from crewai.project import CrewBase, agent, before_kickoff, crew, task
from crewai.task import Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.tools import BaseTool, tool
from crewai.tools.agent_tools.add_image_tool import AddImageTool
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities.rpm_controller import RPMController
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler
from crewai_tools import CodeInterpreterTool
from pydantic import BaseModel, Field
import pydantic_core
import pytest


@pytest.fixture
def ceo():
    return Agent(
        role="CEO",
        goal="Make sure the writers in your company produce amazing content.",
        backstory="You're an long time CEO of a content creation agency with a Senior Writer on the team. You're now working on a new project and want to make sure the content produced is amazing.",
        allow_delegation=True,
    )


@pytest.fixture
def researcher():
    return Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )


@pytest.fixture
def writer():
    return Agent(
        role="Senior Writer",
        goal="Write the best content about AI and AI agents.",
        backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
        allow_delegation=False,
    )


def test_crew_with_only_conditional_tasks_raises_error(researcher):
    """Test that creating a crew with only conditional tasks raises an error."""

    def condition_func(task_output: TaskOutput) -> bool:
        return True

    conditional1 = ConditionalTask(
        description="Conditional task 1",
        expected_output="Output 1",
        agent=researcher,
        condition=condition_func,
    )
    conditional2 = ConditionalTask(
        description="Conditional task 2",
        expected_output="Output 2",
        agent=researcher,
        condition=condition_func,
    )
    conditional3 = ConditionalTask(
        description="Conditional task 3",
        expected_output="Output 3",
        agent=researcher,
        condition=condition_func,
    )

    with pytest.raises(
        pydantic_core._pydantic_core.ValidationError,
        match="Crew must include at least one non-conditional task",
    ):
        Crew(
            agents=[researcher],
            tasks=[conditional1, conditional2, conditional3],
        )


def test_crew_config_conditional_requirement():
    with pytest.raises(ValueError):
        Crew(process=Process.sequential)

    config = json.dumps(
        {
            "agents": [
                {
                    "role": "Senior Researcher",
                    "goal": "Make the best research and analysis on content about AI and AI agents",
                    "backstory": "You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
                },
                {
                    "role": "Senior Writer",
                    "goal": "Write the best content about AI and AI agents.",
                    "backstory": "You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
                },
            ],
            "tasks": [
                {
                    "description": "Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
                    "expected_output": "Bullet point list of 5 important events.",
                    "agent": "Senior Researcher",
                },
                {
                    "description": "Write a 1 amazing paragraph highlight for each idea that showcases how good an article about this topic could be, check references if necessary or search for more content but make sure it's unique, interesting and well written. Return the list of ideas with their paragraph and your notes.",
                    "expected_output": "A 4 paragraph article about AI.",
                    "agent": "Senior Writer",
                },
            ],
        }
    )
    parsed_config = json.loads(config)

    try:
        crew = Crew(process=Process.sequential, config=config)
    except ValueError:
        pytest.fail("Unexpected ValidationError raised")

    assert [agent.role for agent in crew.agents] == [
        agent["role"] for agent in parsed_config["agents"]
    ]
    assert [task.description for task in crew.tasks] == [
        task["description"] for task in parsed_config["tasks"]
    ]


def test_async_task_cannot_include_sequential_async_tasks_in_context(
    researcher, writer
):
    task1 = Task(
        description="Task 1",
        async_execution=True,
        expected_output="output",
        agent=researcher,
    )
    task2 = Task(
        description="Task 2",
        async_execution=True,
        expected_output="output",
        agent=researcher,
        context=[task1],
    )
    task3 = Task(
        description="Task 3",
        async_execution=True,
        expected_output="output",
        agent=researcher,
        context=[task2],
    )
    task4 = Task(
        description="Task 4",
        expected_output="output",
        agent=writer,
    )
    task5 = Task(
        description="Task 5",
        async_execution=True,
        expected_output="output",
        agent=researcher,
        context=[task4],
    )

    # This should raise an error because task2 is async and has task1 in its context without a sync task in between
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Task 'Task 2' is asynchronous and cannot include other sequential asynchronous tasks in its context."
        ),
    ):
        Crew(tasks=[task1, task2, task3, task4, task5], agents=[researcher, writer])

    # This should not raise an error because task5 has a sync task (task4) in its context
    try:
        Crew(tasks=[task1, task4, task5], agents=[researcher, writer])
    except ValueError:
        pytest.fail("Unexpected ValidationError raised")


def test_context_no_future_tasks(researcher, writer):
    task2 = Task(
        description="Task 2",
        expected_output="output",
        agent=researcher,
    )
    task3 = Task(
        description="Task 3",
        expected_output="output",
        agent=researcher,
        context=[task2],
    )
    task4 = Task(
        description="Task 4",
        expected_output="output",
        agent=researcher,
    )
    task1 = Task(
        description="Task 1",
        expected_output="output",
        agent=researcher,
        context=[task4],
    )

    # This should raise an error because task1 has a context dependency on a future task (task4)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Task 'Task 1' has a context dependency on a future task 'Task 4', which is not allowed."
        ),
    ):
        Crew(tasks=[task1, task2, task3, task4], agents=[researcher, writer])


def test_crew_config_with_wrong_keys():
    no_tasks_config = json.dumps(
        {
            "agents": [
                {
                    "role": "Senior Researcher",
                    "goal": "Make the best research and analysis on content about AI and AI agents",
                    "backstory": "You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
                }
            ]
        }
    )

    no_agents_config = json.dumps(
        {
            "tasks": [
                {
                    "description": "Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
                    "agent": "Senior Researcher",
                }
            ]
        }
    )
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config='{"wrong_key": "wrong_value"}')
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config=no_tasks_config)
    with pytest.raises(ValueError):
        Crew(process=Process.sequential, config=no_agents_config)


@pytest.mark.vcr()
def test_crew_creation(researcher, writer):
    tasks = [
        Task(
            description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
            expected_output="Bullet point list of 5 important events.",
            agent=researcher,
        ),
        Task(
            description="Write a 1 amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
            expected_output="A 4 paragraph article about AI.",
            agent=writer,
        ),
    ]

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    result = crew.kickoff()

    expected_string_output = "**The Rise of Generalist AI Agents:**\nImagine a future where AI agents are no longer confined to specific tasks like data analytics or speech recognition. The evolution from specialized AI tools to versatile generalist AI agents is comparable to the leap from feature phones to smartphones. This shift heralds significant transformations across diverse industries, from healthcare and finance to customer service. It also raises fascinating ethical considerations around the deployment and control of such powerful technologies. Moreover, this transformation could democratize AI, making sophisticated tools accessible to non-experts and small businesses, thus leveling the playing field in many sectors.\n\n**Ethical Implications of AI in Surveillance:**\nThe advent of advanced AI has significantly boosted surveillance capabilities, presenting a double-edged sword. On one hand, enhanced surveillance can improve public safety and combat crime more effectively. On the other, it raises substantial ethical concerns about privacy invasion and the potential for misuse by authoritarian regimes. Balancing security with privacy is a delicate task, requiring robust legal frameworks and transparent policies. Real-world case studies, from smart city deployments to airport security systems, illustrate both the benefits and the risks of AI-enhanced surveillance, highlighting the need for ethical vigilance and public discourse.\n\n**AI in Creative Industries:**\nAI is breaking new ground in creative fields, transforming how art, music, and content are produced. Far from being mere tools, AI systems are emerging as collaborators, helping artists push the boundaries of creative expression. Noteworthy are AI-generated works that have captured public imagination, like paintings auctioned at prestigious houses or music albums composed by algorithms. The future holds exciting possibilities, as AI may enable novel art forms and interactive experiences previously unimaginable, fostering a symbiotic relationship between human creativity and machine intelligence.\n\n**The Impact of Quantum Computing on AI Development:**\nQuantum computing promises to be a game-changer for AI, offering unprecedented computational power to tackle complex problems. This revolution could significantly enhance AI algorithms, enabling faster and more efficient training and execution. The potential applications are vast, from optimizing supply chains to solving intricate scientific problems and advancing natural language processing. Looking ahead, quantum-enhanced AI might unlock new frontiers, such as real-time data analysis at scales previously thought impossible, pushing the limits of what we can achieve with AI technology.\n\n**AI and Mental Health:**\nThe integration of AI into mental health care is transforming diagnosis and therapy, offering new hope for those in need. AI-driven tools have shown promise in accurately diagnosing conditions and providing personalized treatment plans through data analysis and pattern recognition. Case studies highlight successful interventions where AI has aided mental health professionals, enhancing the effectiveness of traditional therapies. However, this advancement brings ethical concerns, particularly around data privacy and the transparency of AI decision-making processes. As AI continues to evolve, it could play an even more significant role in mental health care, providing early interventions and support on a scale previously unattainable."

    assert str(result) == expected_string_output
    assert result.raw == expected_string_output
    assert isinstance(result, CrewOutput)
    assert len(result.tasks_output) == len(tasks)
    assert result.raw == expected_string_output


@pytest.mark.vcr()
def test_sync_task_execution(researcher, writer):
    tasks = [
        Task(
            description="Give me a list of 5 interesting ideas to explore for an article, what makes them unique and interesting.",
            expected_output="Bullet point list of 5 important events.",
            agent=researcher,
        ),
        Task(
            description="Write an amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
            expected_output="A 4 paragraph article about AI.",
            agent=writer,
        ),
    ]

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Because we are mocking execute_sync, we never hit the underlying _execute_core
    # which sets the output attribute of the task
    for task in tasks:
        task.output = mock_task_output

    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Assert that execute_sync was called for each task
        assert mock_execute_sync.call_count == len(tasks)


@pytest.mark.vcr()
def test_hierarchical_process(researcher, writer):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        tasks=[task],
    )

    result = crew.kickoff()

    assert (
        result.raw
        == "**1. The Rise of Autonomous AI Agents in Daily Life**  \nAs artificial intelligence technology progresses, the integration of autonomous AI agents into everyday life becomes increasingly prominent. These agents, capable of making decisions without human intervention, are reshaping industries from healthcare to finance. Exploring case studies where autonomous AI has successfully decreased operational costs or improved efficiency can reveal not only the benefits but also the ethical implications of delegating decision-making to machines. This topic offers an exciting opportunity to dive into the AI landscape, showcasing current developments such as AI assistants and autonomous vehicles.\n\n**2. Ethical Implications of Generative AI in Creative Industries**  \nThe surge of generative AI tools in creative fields, such as art, music, and writing, has sparked a heated debate about authorship and originality. This article could investigate how these tools are being used by artists and creators, examining both the potential for innovation and the risk of devaluing traditional art forms. Highlighting perspectives from creators, legal experts, and ethicists could provide a comprehensive overview of the challenges faced, including copyright concerns and the emotional impact on human artists. This discussion is vital as the creative landscape evolves alongside technological advancements, making it ripe for exploration.\n\n**3. AI in Climate Change Mitigation: Current Solutions and Future Potential**  \nAs the world grapples with climate change, AI technology is increasingly being harnessed to develop innovative solutions for sustainability. From predictive analytics that optimize energy consumption to machine learning algorithms that improve carbon capture methods, AI's potential in environmental science is vast. This topic invites an exploration of existing AI applications in climate initiatives, with a focus on groundbreaking research and initiatives aimed at reducing humanity's carbon footprint. Highlighting successful projects and technology partnerships can illustrate the positive impact AI can have on global climate efforts, inspiring further exploration and investment in this area.\n\n**4. The Future of Work: How AI is Reshaping Employment Landscapes**  \nThe discussions around AI's impact on the workforce are both urgent and complex, as advances in automation and machine learning continue to transform the job market. This article could delve into the current trends of AI-driven job displacement alongside opportunities for upskilling and the creation of new job roles. By examining case studies of companies that integrate AI effectively and the resulting workforce adaptations, readers can gain valuable insights into preparing for a future where humans and AI collaborate. This exploration highlights the importance of policies that promote workforce resilience in the face of change.\n\n**5. Decentralized AI: Exploring the Role of Blockchain in AI Development**  \nAs blockchain technology sweeps through various sectors, its application in AI development presents a fascinating topic worth examining. Decentralized AI could address issues of data privacy, security, and democratization in AI models by allowing users to retain ownership of data while benefiting from AI's capabilities. This article could analyze how decentralized networks are disrupting traditional AI development models, featuring innovative projects that harness the synergy between blockchain and AI. Highlighting potential pitfalls and the future landscape of decentralized AI could stimulate discussion among technologists, entrepreneurs, and policymakers alike.\n\nThese topics not only reflect current trends but also probe deeper into ethical and practical considerations, making them timely and relevant for contemporary audiences."
    )


def test_manager_llm_requirement_for_hierarchical_process(researcher, writer):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
    )

    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        Crew(
            agents=[researcher, writer],
            process=Process.hierarchical,
            tasks=[task],
        )


@pytest.mark.vcr()
def test_manager_agent_delegating_to_assigned_task_agent(researcher, writer):
    """
    Test that the manager agent delegates to the assigned task agent.
    """
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        tasks=[task],
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Because we are mocking execute_sync, we never hit the underlying _execute_core
    # which sets the output attribute of the task
    task.output = mock_task_output

    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Verify execute_sync was called once
        mock_execute_sync.assert_called_once()

        # Get the tools argument from the call
        _, kwargs = mock_execute_sync.call_args
        tools = kwargs["tools"]

        # Verify the delegation tools were passed correctly
        assert len(tools) == 2
        assert any(
            "Delegate a specific task to one of the following coworkers: Researcher"
            in tool.description
            for tool in tools
        )
        assert any(
            "Ask a specific question to one of the following coworkers: Researcher"
            in tool.description
            for tool in tools
        )


@pytest.mark.vcr()
def test_manager_agent_delegating_to_all_agents(researcher, writer):
    """
    Test that the manager agent delegates to all agents when none are specified.
    """
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        tasks=[task],
    )

    crew.kickoff()

    assert crew.manager_agent is not None
    assert crew.manager_agent.tools is not None

    assert len(crew.manager_agent.tools) == 2
    assert (
        "Delegate a specific task to one of the following coworkers: Researcher, Senior Writer\n"
        in crew.manager_agent.tools[0].description
    )
    assert (
        "Ask a specific question to one of the following coworkers: Researcher, Senior Writer\n"
        in crew.manager_agent.tools[1].description
    )


@pytest.mark.vcr()
def test_manager_agent_delegates_with_varied_role_cases():
    """
    Test that the manager agent can delegate to agents regardless of case or whitespace variations in role names.
    This test verifies the fix for issue #1503 where role matching was too strict.
    """
    # Create agents with varied case and whitespace in roles
    researcher_spaced = Agent(
        role=" Researcher ",  # Extra spaces
        goal="Research with spaces in role",
        backstory="A researcher with spaces in role name",
        allow_delegation=False,
    )

    writer_caps = Agent(
        role="SENIOR WRITER",  # All caps
        goal="Write with caps in role",
        backstory="A writer with caps in role name",
        allow_delegation=False,
    )

    task = Task(
        description="Research and write about AI. The researcher should do the research, and the writer should write it up.",
        expected_output="A well-researched article about AI.",
        agent=researcher_spaced,  # Assign to researcher with spaces
    )

    crew = Crew(
        agents=[researcher_spaced, writer_caps],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        tasks=[task],
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )
    task.output = mock_task_output

    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Verify execute_sync was called once
        mock_execute_sync.assert_called_once()

        # Get the tools argument from the call
        _, kwargs = mock_execute_sync.call_args
        tools = kwargs["tools"]

        # Verify the delegation tools were passed correctly and can handle case/whitespace variations
        assert len(tools) == 2

        # Check delegation tool descriptions (should work despite case/whitespace differences)
        delegation_tool = tools[0]
        question_tool = tools[1]

        assert (
            "Delegate a specific task to one of the following coworkers:"
            in delegation_tool.description
        )
        assert (
            " Researcher " in delegation_tool.description
            or "SENIOR WRITER" in delegation_tool.description
        )

        assert (
            "Ask a specific question to one of the following coworkers:"
            in question_tool.description
        )
        assert (
            " Researcher " in question_tool.description
            or "SENIOR WRITER" in question_tool.description
        )


@pytest.mark.vcr()
def test_crew_with_delegating_agents(ceo, writer):
    tasks = [
        Task(
            description="Produce and amazing 1 paragraph draft of an article about AI Agents.",
            expected_output="A 4 paragraph article about AI.",
            agent=ceo,
        )
    ]

    crew = Crew(
        agents=[ceo, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    result = crew.kickoff()

    assert (
        result.raw
        == "In the rapidly evolving landscape of technology, AI agents have emerged as formidable tools, revolutionizing how we interact with data and automate tasks. These sophisticated systems leverage machine learning and natural language processing to perform a myriad of functions, from virtual personal assistants to complex decision-making companions in industries such as finance, healthcare, and education. By mimicking human intelligence, AI agents can analyze massive data sets at unparalleled speeds, enabling businesses to uncover valuable insights, enhance productivity, and elevate user experiences to unprecedented levels.\n\nOne of the most striking aspects of AI agents is their adaptability; they learn from their interactions and continuously improve their performance over time. This feature is particularly valuable in customer service where AI agents can address inquiries, resolve issues, and provide personalized recommendations without the limitations of human fatigue. Moreover, with intuitive interfaces, AI agents enhance user interactions, making technology more accessible and user-friendly, thereby breaking down barriers that have historically hindered digital engagement.\n\nDespite their immense potential, the deployment of AI agents raises important ethical and practical considerations. Issues related to privacy, data security, and the potential for job displacement necessitate thoughtful dialogue and proactive measures. Striking a balance between technological innovation and societal impact will be crucial as organizations integrate these agents into their operations. Additionally, ensuring transparency in AI decision-making processes is vital to maintain public trust as AI agents become an integral part of daily life.\n\nLooking ahead, the future of AI agents appears bright, with ongoing advancements promising even greater capabilities. As we continue to harness the power of AI, we can expect these agents to play a transformative role in shaping various sectors—streamlining workflows, enabling smarter decision-making, and fostering more personalized experiences. Embracing this technology responsibly can lead to a future where AI agents not only augment human effort but also inspire creativity and efficiency across the board, ultimately redefining our interaction with the digital world."
    )


@pytest.mark.vcr()
def test_crew_with_delegating_agents_should_not_override_task_tools(ceo, writer):
    class TestToolInput(BaseModel):
        """Input schema for TestTool."""

        query: str = Field(..., description="Query to process")

    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool that just returns the input"
        args_schema: type[BaseModel] = TestToolInput

        def _run(self, query: str) -> str:
            return f"Processed: {query}"

    # Create a task with the test tool
    tasks = [
        Task(
            description="Produce and amazing 1 paragraph draft of an article about AI Agents.",
            expected_output="A 4 paragraph article about AI.",
            agent=ceo,
            tools=[TestTool()],
        )
    ]

    crew = Crew(
        agents=[ceo, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Because we are mocking execute_sync, we never hit the underlying _execute_core
    # which sets the output attribute of the task
    tasks[0].output = mock_task_output

    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Execute the task and verify both tools are present
        _, kwargs = mock_execute_sync.call_args
        tools = kwargs["tools"]

        assert any(isinstance(tool, TestTool) for tool in tools), (
            "TestTool should be present"
        )
        assert any("delegate" in tool.name.lower() for tool in tools), (
            "Delegation tool should be present"
        )


@pytest.mark.vcr()
def test_crew_with_delegating_agents_should_not_override_agent_tools(ceo, writer):
    class TestToolInput(BaseModel):
        """Input schema for TestTool."""

        query: str = Field(..., description="Query to process")

    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool that just returns the input"
        args_schema: type[BaseModel] = TestToolInput

        def _run(self, query: str) -> str:
            return f"Processed: {query}"

    new_ceo = ceo.model_copy()
    new_ceo.tools = [TestTool()]

    # Create a task with the test tool
    tasks = [
        Task(
            description="Produce and amazing 1 paragraph draft of an article about AI Agents.",
            expected_output="A 4 paragraph article about AI.",
            agent=new_ceo,
        )
    ]

    crew = Crew(
        agents=[new_ceo, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Because we are mocking execute_sync, we never hit the underlying _execute_core
    # which sets the output attribute of the task
    tasks[0].output = mock_task_output

    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Execute the task and verify both tools are present
        _, kwargs = mock_execute_sync.call_args
        tools = kwargs["tools"]

        assert any(isinstance(tool, TestTool) for tool in new_ceo.tools), (
            "TestTool should be present"
        )
        assert any("delegate" in tool.name.lower() for tool in tools), (
            "Delegation tool should be present"
        )


@pytest.mark.vcr()
def test_task_tools_override_agent_tools(researcher):
    class TestToolInput(BaseModel):
        """Input schema for TestTool."""

        query: str = Field(..., description="Query to process")

    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool that just returns the input"
        args_schema: type[BaseModel] = TestToolInput

        def _run(self, query: str) -> str:
            return f"Processed: {query}"

    class AnotherTestTool(BaseTool):
        name: str = "Another Test Tool"
        description: str = "Another test tool"
        args_schema: type[BaseModel] = TestToolInput

        def _run(self, query: str) -> str:
            return f"Another processed: {query}"

    # Set agent tools
    new_researcher = researcher.model_copy()
    new_researcher.tools = [TestTool()]

    # Create task with different tools
    task = Task(
        description="Write a test task",
        expected_output="Test output",
        agent=new_researcher,
        tools=[AnotherTestTool()],
    )

    crew = Crew(agents=[new_researcher], tasks=[task], process=Process.sequential)

    crew.kickoff()

    # Verify task tools override agent tools
    assert len(task.tools) == 1  # AnotherTestTool
    assert any(isinstance(tool, AnotherTestTool) for tool in task.tools)
    assert not any(isinstance(tool, TestTool) for tool in task.tools)

    # Verify agent tools remain unchanged
    assert len(new_researcher.tools) == 1
    assert isinstance(new_researcher.tools[0], TestTool)


@pytest.mark.vcr()
def test_task_tools_override_agent_tools_with_allow_delegation(researcher, writer):
    """
    Test that task tools override agent tools while preserving delegation tools when allow_delegation=True
    """
    class TestToolInput(BaseModel):
        query: str = Field(..., description="Query to process")

    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool that just returns the input"
        args_schema: type[BaseModel] = TestToolInput

        def _run(self, query: str) -> str:
            return f"Processed: {query}"

    class AnotherTestTool(BaseTool):
        name: str = "Another Test Tool"
        description: str = "Another test tool"
        args_schema: type[BaseModel] = TestToolInput

        def _run(self, query: str) -> str:
            return f"Another processed: {query}"

    # Set up agents with tools and allow_delegation
    researcher_with_delegation = researcher.model_copy()
    researcher_with_delegation.allow_delegation = True
    researcher_with_delegation.tools = [TestTool()]

    writer_for_delegation = writer.model_copy()

    # Create a task with different tools
    task = Task(
        description="Write a test task",
        expected_output="Test output",
        agent=researcher_with_delegation,
        tools=[AnotherTestTool()],
    )

    crew = Crew(
        agents=[researcher_with_delegation, writer_for_delegation],
        tasks=[task],
        process=Process.sequential,
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # We mock execute_sync to verify which tools get used at runtime
    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Inspect the call kwargs to verify the actual tools passed to execution
        _, kwargs = mock_execute_sync.call_args
        used_tools = kwargs["tools"]

        # Confirm AnotherTestTool is present but TestTool is not
        assert any(isinstance(tool, AnotherTestTool) for tool in used_tools), (
            "AnotherTestTool should be present"
        )
        assert not any(isinstance(tool, TestTool) for tool in used_tools), (
            "TestTool should not be present among used tools"
        )

        # Confirm delegation tool(s) are present
        assert any("delegate" in tool.name.lower() for tool in used_tools), (
            "Delegation tool should be present"
        )

    # Finally, make sure the agent's original tools remain unchanged
    assert len(researcher_with_delegation.tools) == 1
    assert isinstance(researcher_with_delegation.tools[0], TestTool)


@pytest.mark.vcr()
def test_crew_verbose_output(researcher, writer, capsys):
    tasks = [
        Task(
            description="Research AI advancements.",
            expected_output="A full report on AI advancements.",
            agent=researcher,
        ),
        Task(
            description="Write about AI in healthcare.",
            expected_output="A 4 paragraph article about AI.",
            agent=writer,
        ),
    ]

    # Test with verbose=True
    crew = Crew(
        agents=[researcher, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    # Verify the crew executed successfully and verbose was set
    assert result is not None
    assert crew.verbose is True

    # Test with verbose=False
    crew_quiet = Crew(
        agents=[researcher, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    result_quiet = crew_quiet.kickoff()

    # Verify the crew executed successfully and verbose was not set
    assert result_quiet is not None
    assert crew_quiet.verbose is False


@pytest.mark.vcr()
def test_cache_hitting_between_agents(researcher, writer, ceo):
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    tasks = [
        Task(
            description="What is 2 tims 6? Return only the number.",
            expected_output="the result of multiplication",
            tools=[multiplier],
            agent=ceo,
        ),
        Task(
            description="What is 2 times 6? Return only the number.",
            expected_output="the result of multiplication",
            tools=[multiplier],
            agent=researcher,
        ),
    ]

    crew = Crew(
        agents=[ceo, researcher],
        tasks=tasks,
    )

    with patch.object(CacheHandler, "read") as read:
        read.return_value = "12"
        crew.kickoff()
        assert read.call_count == 2, "read was not called exactly twice"

        # Filter the mock calls to only include the ones with 'tool' and 'input' keywords
        cache_calls = [
            call
            for call in read.call_args_list
            if len(call.kwargs) == 2
            and "tool" in call.kwargs
            and "input" in call.kwargs
        ]

        # Check if we have the expected number of cache calls
        assert len(cache_calls) == 2, f"Expected 2 cache calls, got {len(cache_calls)}"

        # Check if both calls were made with the expected arguments
        expected_call = call(
            tool="multiplier", input='{"first_number": 2, "second_number": 6}'
        )
        assert cache_calls[0] == expected_call, f"First call mismatch: {cache_calls[0]}"
        assert cache_calls[1] == expected_call, (
            f"Second call mismatch: {cache_calls[1]}"
        )


@pytest.mark.vcr()
def test_api_calls_throttling(capsys):
    @tool
    def get_final_answer() -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent = Agent(
        role="Very helpful assistant",
        goal="Comply with necessary changes",
        backstory="You obey orders",
        max_iter=2,
        allow_delegation=False,
        verbose=True,
        llm="gpt-4o",
    )

    task = Task(
        description="Don't give a Final Answer unless explicitly told it's time to give the absolute best final answer.",
        expected_output="The final answer.",
        tools=[get_final_answer],
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], max_rpm=1, verbose=True)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called()


@pytest.mark.vcr()
def test_crew_kickoff_usage_metrics():
    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
        llm=LLM(model="gpt-4o"),
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    # Use real LLM calls instead of mocking
    crew = Crew(agents=[agent], tasks=[task])
    results = crew.kickoff_for_each(inputs=inputs)

    assert len(results) == len(inputs)
    for result in results:
        # Assert that all required keys are in usage_metrics and their values are greater than 0
        assert result.token_usage.total_tokens > 0
        assert result.token_usage.prompt_tokens > 0
        assert result.token_usage.completion_tokens > 0
        assert result.token_usage.successful_requests > 0
        assert result.token_usage.cached_prompt_tokens == 0


@pytest.mark.vcr()
def test_crew_kickoff_streaming_usage_metrics():
    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
        llm=LLM(model="gpt-4o", stream=True, is_litellm=True),
        max_iter=3,
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    # Use real LLM calls instead of mocking
    crew = Crew(agents=[agent], tasks=[task])
    results = crew.kickoff_for_each(inputs=inputs)

    assert len(results) == len(inputs)
    for result in results:
        # Assert that all required keys are in usage_metrics and their values are greater than 0
        assert result.token_usage.total_tokens > 0
        assert result.token_usage.prompt_tokens > 0
        assert result.token_usage.completion_tokens > 0
        assert result.token_usage.successful_requests > 0
        assert result.token_usage.cached_prompt_tokens == 0


def test_agents_rpm_is_never_set_if_crew_max_rpm_is_not_set():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description="just say hi!",
        expected_output="your greeting",
        agent=agent,
    )

    Crew(agents=[agent], tasks=[task], verbose=True)

    assert agent._rpm_controller is None


@pytest.mark.vcr()
def test_sequential_async_task_execution_completion(researcher, writer):
    list_ideas = Task(
        description="Give me a list of 5 interesting ideas to explore for an article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher,
        async_execution=True,
    )
    list_important_history = Task(
        description="Research the history of AI and give me the 5 most important events that shaped the technology.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher,
    )
    write_article = Task(
        description="Write an article about the history of AI and its most important events.",
        expected_output="A 4 paragraph article about AI.",
        agent=writer,
        context=[list_ideas, list_important_history],
    )

    sequential_crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[list_ideas, list_important_history, write_article],
    )

    sequential_result = sequential_crew.kickoff()
    assert sequential_result.raw.startswith(
        "The history of artificial intelligence (AI) is marked by several pivotal events that have shaped the field into what it is today."
    )


@pytest.mark.vcr()
def test_single_task_with_async_execution():
    researcher_agent = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    list_ideas = Task(
        description="Generate a list of 5 interesting ideas to explore for an article, where each bulletpoint is under 15 words.",
        expected_output="Bullet point list of 5 important events. No additional commentary.",
        agent=researcher_agent,
        async_execution=True,
    )

    crew = Crew(
        agents=[researcher_agent],
        process=Process.sequential,
        tasks=[list_ideas],
    )

    result = crew.kickoff()
    assert result.raw.startswith(
        "- Ethical implications of AI in law enforcement and surveillance."
    )


@pytest.mark.vcr()
def test_three_task_with_async_execution():
    researcher_agent = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    bullet_list = Task(
        description="Generate a list of 5 interesting ideas to explore for an article, where each bulletpoint is under 15 words.",
        expected_output="Bullet point list of 5 important events. No additional commentary.",
        agent=researcher_agent,
        async_execution=True,
    )
    numbered_list = Task(
        description="Generate a list of 5 interesting ideas to explore for an article, where each bulletpoint is under 15 words.",
        expected_output="Numbered list of 5 important events. No additional commentary.",
        agent=researcher_agent,
        async_execution=True,
    )
    letter_list = Task(
        description="Generate a list of 5 interesting ideas to explore for an article, where each bulletpoint is under 15 words.",
        expected_output="Numbered list using [A), B), C)] list of 5 important events. No additional commentary.",
        agent=researcher_agent,
        async_execution=True,
    )

    # Expected result is that we will get an error
    # because a crew can end only end with one or less
    # async tasks
    with pytest.raises(pydantic_core._pydantic_core.ValidationError) as error:
        Crew(
            agents=[researcher_agent],
            process=Process.sequential,
            tasks=[bullet_list, numbered_list, letter_list],
        )

    assert error.value.errors()[0]["type"] == "async_task_count"
    assert (
        "The crew must end with at most one asynchronous task."
        in error.value.errors()[0]["msg"]
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_crew_async_kickoff():
    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]

    agent = Agent(
        role="mock agent",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    mock_task_output = (
        CrewOutput(
            raw="Test output from Crew 1",
            tasks_output=[],
            token_usage=UsageMetrics(
                total_tokens=100,
                prompt_tokens=10,
                completion_tokens=90,
                successful_requests=1,
            ),
            json_dict={"output": "crew1"},
            pydantic=None,
        ),
    )
    with patch.object(Crew, "kickoff_async", return_value=mock_task_output):
        results = await crew.kickoff_for_each_async(inputs=inputs)

        assert len(results) == len(inputs)
        for result in results:
            # Assert that all required keys are in usage_metrics and their values are not None
            assert result[0].token_usage.total_tokens > 0  # type: ignore
            assert result[0].token_usage.prompt_tokens > 0  # type: ignore
            assert result[0].token_usage.completion_tokens > 0  # type: ignore
            assert result[0].token_usage.successful_requests > 0  # type: ignore


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_task_execution_call_count(researcher, writer):
    list_ideas = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher,
        async_execution=True,
    )
    list_important_history = Task(
        description="Research the history of AI and give me the 5 most important events that shaped the technology.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher,
        async_execution=True,
    )
    write_article = Task(
        description="Write an article about the history of AI and its most important events.",
        expected_output="A 4 paragraph article about AI.",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[list_ideas, list_important_history, write_article],
    )

    # Create a valid TaskOutput instance to mock the return value
    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Create a MagicMock Future instance
    mock_future = MagicMock(spec=Future)
    mock_future.result.return_value = mock_task_output

    # Directly set the output attribute for each task
    list_ideas.output = mock_task_output
    list_important_history.output = mock_task_output
    write_article.output = mock_task_output

    with (
        patch.object(
            Task, "execute_sync", return_value=mock_task_output
        ) as mock_execute_sync,
        patch.object(
            Task, "execute_async", return_value=mock_future
        ) as mock_execute_async,
    ):
        crew.kickoff()

        assert mock_execute_async.call_count == 2
        assert mock_execute_sync.call_count == 1


@pytest.mark.vcr()
def test_kickoff_for_each_single_input():
    """Tests if kickoff_for_each works with a single input."""

    inputs = [{"topic": "dog"}]

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    results = crew.kickoff_for_each(inputs=inputs)

    assert len(results) == 1


@pytest.mark.vcr()
def test_kickoff_for_each_multiple_inputs():
    """Tests if kickoff_for_each works with multiple inputs."""

    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    results = crew.kickoff_for_each(inputs=inputs)

    assert len(results) == len(inputs)


@pytest.mark.vcr()
def test_kickoff_for_each_empty_input():
    """Tests if kickoff_for_each handles an empty input list."""
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    results = crew.kickoff_for_each(inputs=[])
    assert results == []


@pytest.mark.vcr()
def test_kickoff_for_each_invalid_input():
    """Tests if kickoff_for_each raises TypeError for invalid input types."""

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])

    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        # Pass a string instead of a list
        crew.kickoff_for_each(["invalid input"])


def test_kickoff_for_each_error_handling():
    """Tests error handling in kickoff_for_each when kickoff raises an error."""

    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]
    expected_outputs = [
        "Dogs are loyal companions and popular pets.",
        "Cats are independent and low-maintenance pets.",
        "Apples are a rich source of dietary fiber and vitamin C.",
    ]
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])

    with patch.object(Crew, "kickoff") as mock_kickoff:
        mock_kickoff.side_effect = [
            *expected_outputs[:2],
            Exception("Simulated kickoff error"),
        ]
        with pytest.raises(Exception, match="Simulated kickoff error"):
            crew.kickoff_for_each(inputs=inputs)


@pytest.mark.asyncio
async def test_kickoff_async_basic_functionality_and_output():
    """Tests the basic functionality and output of kickoff_async."""

    inputs = {"topic": "dog"}

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    # Create the crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
    )

    expected_output = "This is a sample output from kickoff."
    with patch.object(Crew, "kickoff", return_value=expected_output) as mock_kickoff:
        result = await crew.kickoff_async(inputs)

        assert isinstance(result, str), "Result should be a string"
        assert result == expected_output, "Result should match expected output"
        mock_kickoff.assert_called_once_with(inputs)


@pytest.mark.asyncio
async def test_async_kickoff_for_each_async_basic_functionality_and_output():
    """Tests the basic functionality and output of kickoff_for_each_async."""
    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]

    # Define expected outputs for each input
    expected_outputs = [
        "Dogs are loyal companions and popular pets.",
        "Cats are independent and low-maintenance pets.",
        "Apples are a rich source of dietary fiber and vitamin C.",
    ]

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    async def mock_kickoff_async(**kwargs):
        input_data = kwargs.get("inputs")
        index = [input_["topic"] for input_ in inputs].index(input_data["topic"])
        return expected_outputs[index]

    with patch.object(
        Crew, "kickoff_async", side_effect=mock_kickoff_async
    ) as mock_kickoff_async:
        crew = Crew(agents=[agent], tasks=[task])

        results = await crew.kickoff_for_each_async(inputs)

        assert len(results) == len(inputs)
        assert results == expected_outputs
        for input_data in inputs:
            mock_kickoff_async.assert_any_call(inputs=input_data)


@pytest.mark.asyncio
async def test_async_kickoff_for_each_async_empty_input():
    """Tests if akickoff_for_each_async handles an empty input list."""

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )

    # Create the crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
    )

    # Call the function we are testing
    results = await crew.kickoff_for_each_async([])

    # Assertion
    assert results == [], "Result should be an empty list when input is empty"


def test_set_agents_step_callback():
    researcher_agent = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    list_ideas = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher_agent,
        async_execution=True,
    )

    crew = Crew(
        agents=[researcher_agent],
        process=Process.sequential,
        tasks=[list_ideas],
        step_callback=lambda: None,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        crew.kickoff()
        assert researcher_agent.step_callback is not None


def test_dont_set_agents_step_callback_if_already_set():
    def agent_callback(_):
        pass

    def crew_callback(_):
        pass

    researcher_agent = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
        step_callback=agent_callback,
    )

    list_ideas = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher_agent,
        async_execution=True,
    )

    crew = Crew(
        agents=[researcher_agent],
        process=Process.sequential,
        tasks=[list_ideas],
        step_callback=crew_callback,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        crew.kickoff()
        assert researcher_agent.step_callback is not crew_callback
        assert researcher_agent.step_callback is agent_callback


@pytest.mark.vcr()
def test_crew_function_calling_llm():
    llm = LLM(model="gpt-4o-mini")

    @tool
    def look_up_greeting() -> str:
        """Tool used to retrieve a greeting."""
        return "Howdy!"

    agent1 = Agent(
        role="Greeter",
        goal="Say hello.",
        backstory="You are a friendly greeter.",
        tools=[look_up_greeting],
        llm="gpt-4o-mini",
        function_calling_llm=llm,
    )

    essay = Task(
        description="Look up the greeting and say it.",
        expected_output="A greeting.",
        agent=agent1,
    )

    crew = Crew(agents=[agent1], tasks=[essay])
    result = crew.kickoff()
    assert result.raw == "Howdy!"


@pytest.mark.vcr()
def test_task_with_no_arguments():
    @tool
    def return_data() -> str:
        "Useful to get the sales related data"
        return "January: 5, February: 10, March: 15, April: 20, May: 25"

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        tools=[return_data],
        allow_delegation=False,
    )

    task = Task(
        description="Look at the available data and give me a sense on the total number of sales.",
        expected_output="The total number of sales as an integer",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task])

    result = crew.kickoff()
    assert result.raw == "The total number of sales is 75."


def test_code_execution_flag_adds_code_tool_upon_kickoff():
    # Mock Docker validation for the entire test
    with patch.object(Agent, "_validate_docker_installation"):
        programmer = Agent(
            role="Programmer",
            goal="Write code to solve problems.",
            backstory="You're a programmer who loves to solve problems with code.",
            allow_delegation=False,
            allow_code_execution=True,
        )

        task = Task(
            description="How much is 2 + 2?",
            expected_output="The result of the sum as an integer.",
            agent=programmer,
        )

        crew = Crew(agents=[programmer], tasks=[task])

        mock_task_output = TaskOutput(
            description="Mock description", raw="mocked output", agent="mocked agent"
        )

        with patch.object(
            Task, "execute_sync", return_value=mock_task_output
        ) as mock_execute_sync:
            crew.kickoff()

            # Get the tools that were actually used in execution
            _, kwargs = mock_execute_sync.call_args
            used_tools = kwargs["tools"]

            # Verify that exactly one tool was used and it was a CodeInterpreterTool
            assert len(used_tools) == 1, "Should have exactly one tool"
            assert isinstance(used_tools[0], CodeInterpreterTool), (
                "Tool should be CodeInterpreterTool"
            )


@pytest.mark.vcr()
def test_delegation_is_not_enabled_if_there_are_only_one_agent():
    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=True,
    )

    task = Task(
        description="Look at the available data and give me a sense on the total number of sales.",
        expected_output="The total number of sales as an integer",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task])

    crew.kickoff()
    assert task.tools == []


@pytest.mark.vcr()
def test_agents_do_not_get_delegation_tools_with_there_is_only_one_agent():
    agent = Agent(
        role="Researcher",
        goal="Be super empathetic.",
        backstory="You're love to sey howdy.",
        allow_delegation=False,
    )

    task = Task(description="say howdy", expected_output="Howdy!", agent=agent)

    crew = Crew(agents=[agent], tasks=[task])

    result = crew.kickoff()
    assert result.raw == "Howdy!"
    assert len(agent.tools) == 0


@pytest.mark.vcr()
def test_sequential_crew_creation_tasks_without_agents(researcher):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        # agent=researcher, # not having an agent on the task should throw an error
    )

    # Expected Output: The sequential crew should fail to create because the task is missing an agent
    with pytest.raises(pydantic_core._pydantic_core.ValidationError) as exec_info:
        Crew(
            tasks=[task],
            agents=[researcher],
            process=Process.sequential,
        )

    assert exec_info.value.errors()[0]["type"] == "missing_agent_in_task"
    assert (
        "Agent is missing in the task with the following description"
        in exec_info.value.errors()[0]["msg"]
    )


@pytest.mark.vcr()
def test_agent_usage_metrics_are_captured_for_hierarchical_process():
    agent = Agent(
        role="Researcher",
        goal="Be super empathetic.",
        backstory="You're love to sey howdy.",
        allow_delegation=False,
    )

    task = Task(description="Ask the researched to say hi!", expected_output="Howdy!")

    crew = Crew(
        agents=[agent], tasks=[task], process=Process.hierarchical, manager_llm="gpt-4o"
    )

    result = crew.kickoff()
    assert result.raw == "Howdy!"

    assert result.token_usage == UsageMetrics(
        total_tokens=1673,
        prompt_tokens=1562,
        completion_tokens=111,
        successful_requests=3,
        cached_prompt_tokens=0,
    )


def test_hierarchical_kickoff_usage_metrics_include_manager(researcher):
    """Ensure Crew.kickoff() sums UsageMetrics from both regular and manager agents."""

    # ── 1.  Build the manager and a simple task ──────────────────────────────────
    manager = Agent(
        role="Manager",
        goal="Coordinate everything.",
        backstory="Keeps the project on track.",
        allow_delegation=False,
    )

    task = Task(
        description="Say hello",
        expected_output="Hello",
        agent=researcher,  # *regular* agent
    )

    # ── 2.  Stub out each agent's token usage methods ───────────────────
    researcher_metrics = UsageMetrics(
        total_tokens=120, prompt_tokens=80, completion_tokens=40, successful_requests=2
    )
    manager_metrics = UsageMetrics(
        total_tokens=30, prompt_tokens=20, completion_tokens=10, successful_requests=1
    )

    # Mock the LLM's get_token_usage_summary method for the researcher
    researcher.llm.get_token_usage_summary = MagicMock(return_value=researcher_metrics)

    # Mock the manager's _token_process since it uses the fallback path
    manager._token_process = MagicMock(
        get_summary=MagicMock(return_value=manager_metrics)
    )

    # ── 3.  Create the crew (hierarchical!) and kick it off ──────────────────────
    crew = Crew(
        agents=[researcher],  # regular agents
        manager_agent=manager,  # manager to be included
        tasks=[task],
        process=Process.hierarchical,
    )

    # We don't care about LLM output here; patch execute_sync to avoid network
    with patch.object(
        Task,
        "execute_sync",
        return_value=TaskOutput(
            description="dummy", raw="Hello", agent=researcher.role, messages=[]
        ),
    ):
        crew.kickoff()

    # ── 4.  Assert the aggregated numbers are the *sum* of both agents ───────────
    assert (
        crew.usage_metrics.total_tokens
        == researcher_metrics.total_tokens + manager_metrics.total_tokens
    )
    assert (
        crew.usage_metrics.prompt_tokens
        == researcher_metrics.prompt_tokens + manager_metrics.prompt_tokens
    )
    assert (
        crew.usage_metrics.completion_tokens
        == researcher_metrics.completion_tokens + manager_metrics.completion_tokens
    )
    assert (
        crew.usage_metrics.successful_requests
        == researcher_metrics.successful_requests + manager_metrics.successful_requests
    )


@pytest.mark.vcr()
def test_hierarchical_crew_creation_tasks_with_agents(researcher, writer):
    """
    Agents are not required for tasks in a hierarchical process but sometimes they are still added
    This test makes sure that the manager still delegates the task to the agent even if the agent is passed in the task
    """
    task = Task(
        description="Write one amazing paragraph about AI.",
        expected_output="A single paragraph with 4 sentences.",
        agent=writer,
    )

    crew = Crew(
        tasks=[task],
        agents=[writer, researcher],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Because we are mocking execute_sync, we never hit the underlying _execute_core
    # which sets the output attribute of the task
    task.output = mock_task_output

    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Verify execute_sync was called once
        mock_execute_sync.assert_called_once()

        # Get the tools argument from the call
        _, kwargs = mock_execute_sync.call_args
        tools = kwargs["tools"]

        # Verify the delegation tools were passed correctly
        assert len(tools) == 2
        assert any(
            "Delegate a specific task to one of the following coworkers: Senior Writer"
            in tool.description
            for tool in tools
        )
        assert any(
            "Ask a specific question to one of the following coworkers: Senior Writer"
            in tool.description
            for tool in tools
        )


@pytest.mark.vcr()
def test_hierarchical_crew_creation_tasks_with_async_execution(researcher, writer, ceo):
    """
    Tests that async tasks in hierarchical crews are handled correctly with proper delegation tools
    """
    task = Task(
        description="Write one amazing paragraph about AI.",
        expected_output="A single paragraph with 4 sentences.",
        agent=writer,
        async_execution=True,
    )

    crew = Crew(
        tasks=[task],
        agents=[writer, researcher, ceo],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Create a mock Future that returns our TaskOutput
    mock_future = MagicMock(spec=Future)
    mock_future.result.return_value = mock_task_output

    # Because we are mocking execute_async, we never hit the underlying _execute_core
    # which sets the output attribute of the task
    task.output = mock_task_output

    with patch.object(
        Task, "execute_async", return_value=mock_future
    ) as mock_execute_async:
        crew.kickoff()

        # Verify execute_async was called once
        mock_execute_async.assert_called_once()

        # Get the tools argument from the call
        _, kwargs = mock_execute_async.call_args
        tools = kwargs["tools"]

        # Verify the delegation tools were passed correctly
        assert len(tools) == 2
        assert any(
            "Delegate a specific task to one of the following coworkers: Senior Writer\n"
            in tool.description
            for tool in tools
        )
        assert any(
            "Ask a specific question to one of the following coworkers: Senior Writer\n"
            in tool.description
            for tool in tools
        )


@pytest.mark.vcr()
def test_hierarchical_crew_creation_tasks_with_sync_last(researcher, writer, ceo):
    """
    Agents are not required for tasks in a hierarchical process but sometimes they are still added
    This test makes sure that the manager still delegates the task to the agent even if the agent is passed in the task
    """
    task = Task(
        description="Write one amazing paragraph about AI.",
        expected_output="A single paragraph with 4 sentences.",
        agent=writer,
        async_execution=True,
    )
    task2 = Task(
        description="Write one amazing paragraph about AI.",
        expected_output="A single paragraph with 4 sentences.",
        async_execution=False,
    )

    crew = Crew(
        tasks=[task, task2],
        agents=[writer, researcher, ceo],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
    )

    crew.kickoff()
    assert crew.manager_agent is not None
    assert crew.manager_agent.tools is not None
    assert (
        "Delegate a specific task to one of the following coworkers: Senior Writer, Researcher, CEO\n"
        in crew.manager_agent.tools[0].description
    )


def test_crew_inputs_interpolate_both_agents_and_tasks():
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="{points} bullet points about {topic}.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    inputs = {"topic": "AI", "points": 5}
    crew._interpolate_inputs(inputs=inputs)  # Manual call for now

    assert crew.tasks[0].description == "Give me an analysis around AI."
    assert crew.tasks[0].expected_output == "5 bullet points about AI."
    assert crew.agents[0].role == "AI Researcher"
    assert crew.agents[0].goal == "Express hot takes on AI."
    assert crew.agents[0].backstory == "You have a lot of experience with AI."


def test_crew_inputs_interpolate_both_agents_and_tasks_diff():
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="{points} bullet points about {topic}.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])

    with patch.object(Agent, "execute_task") as execute:
        with patch.object(
            Agent, "interpolate_inputs", wraps=agent.interpolate_inputs
        ) as interpolate_agent_inputs:
            with patch.object(
                Task,
                "interpolate_inputs_and_add_conversation_history",
                wraps=task.interpolate_inputs_and_add_conversation_history,
            ) as interpolate_task_inputs:
                execute.return_value = "ok"
                crew.kickoff(inputs={"topic": "AI", "points": 5})
                interpolate_agent_inputs.assert_called()
                interpolate_task_inputs.assert_called()


@pytest.mark.vcr()
def test_crew_does_not_interpolate_without_inputs():
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="{points} bullet points about {topic}.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])

    with patch.object(Agent, "interpolate_inputs") as interpolate_agent_inputs:
        with patch.object(
            Task, "interpolate_inputs_and_add_conversation_history"
        ) as interpolate_task_inputs:
            crew.kickoff()
            interpolate_agent_inputs.assert_not_called()
            interpolate_task_inputs.assert_not_called()


def test_task_callback_on_crew():
    researcher_agent = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    list_ideas = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher_agent,
        async_execution=True,
    )

    mock_callback = MagicMock()

    crew = Crew(
        agents=[researcher_agent],
        process=Process.sequential,
        tasks=[list_ideas],
        task_callback=mock_callback,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        crew.kickoff()

        assert list_ideas.callback is not None
        mock_callback.assert_called_once()
        args, _ = mock_callback.call_args
        assert isinstance(args[0], TaskOutput)


def test_task_callback_both_on_task_and_crew():
    mock_callback_on_task = MagicMock()
    mock_callback_on_crew = MagicMock()

    researcher_agent = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    list_ideas = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher_agent,
        async_execution=True,
        callback=mock_callback_on_task,
    )

    crew = Crew(
        agents=[researcher_agent],
        process=Process.sequential,
        tasks=[list_ideas],
        task_callback=mock_callback_on_crew,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        crew.kickoff()

        assert list_ideas.callback is not None
        mock_callback_on_task.assert_called_once_with(list_ideas.output)
        mock_callback_on_crew.assert_called_once_with(list_ideas.output)


def test_task_same_callback_both_on_task_and_crew():
    mock_callback = MagicMock()

    researcher_agent = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    list_ideas = Task(
        description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
        expected_output="Bullet point list of 5 important events.",
        agent=researcher_agent,
        async_execution=True,
        callback=mock_callback,
    )

    crew = Crew(
        agents=[researcher_agent],
        process=Process.sequential,
        tasks=[list_ideas],
        task_callback=mock_callback,
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        crew.kickoff()

        assert list_ideas.callback is not None
        mock_callback.assert_called_once_with(list_ideas.output)


@pytest.mark.vcr()
def test_tools_with_custom_caching():
    @tool
    def multiplcation_tool(first_number: int, second_number: int) -> int:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    def cache_func(args, result):
        return result % 2 == 0

    multiplcation_tool.cache_function = cache_func

    writer1 = Agent(
        role="Writer",
        goal="You write lessons of math for kids.",
        backstory="You're an expert in writing and you love to teach kids but you know nothing of math.",
        tools=[multiplcation_tool],
        allow_delegation=False,
    )

    writer2 = Agent(
        role="Writer",
        goal="You write lessons of math for kids.",
        backstory="You're an expert in writing and you love to teach kids but you know nothing of math.",
        tools=[multiplcation_tool],
        allow_delegation=False,
    )

    task1 = Task(
        description="What is 2 times 6? Return only the number after using the multiplication tool.",
        expected_output="the result of multiplication",
        agent=writer1,
    )

    task2 = Task(
        description="What is 3 times 1? Return only the number after using the multiplication tool.",
        expected_output="the result of multiplication",
        agent=writer1,
    )

    task3 = Task(
        description="What is 2 times 6? Return only the number after using the multiplication tool.",
        expected_output="the result of multiplication",
        agent=writer2,
    )

    task4 = Task(
        description="What is 3 times 1? Return only the number after using the multiplication tool.",
        expected_output="the result of multiplication",
        agent=writer2,
    )

    crew = Crew(agents=[writer1, writer2], tasks=[task1, task2, task3, task4])

    with patch.object(
        CacheHandler, "add", wraps=crew._cache_handler.add
    ) as add_to_cache:
        result = crew.kickoff()

        # Check that add_to_cache was called exactly twice
        assert add_to_cache.call_count == 2

        # Verify that one of those calls was with the even number that should be cached
        add_to_cache.assert_any_call(
            tool="multiplcation_tool",
            input='{"first_number": 2, "second_number": 6}',
            output=12,
        )

        assert result.raw == "3"


@pytest.mark.vcr()
def test_conditional_task_uses_last_output(researcher, writer):
    """Test that conditional tasks use the last task output for condition evaluation."""
    task1 = Task(
        description="First task",
        expected_output="First output",
        agent=researcher,
    )

    def condition_fails(task_output: TaskOutput) -> bool:
        # This condition will never be met
        return "never matches" in task_output.raw.lower()

    def condition_succeeds(task_output: TaskOutput) -> bool:
        # This condition will match first task's output
        return "first success" in task_output.raw.lower()

    conditional_task1 = ConditionalTask(
        description="Second task - conditional that fails condition",
        expected_output="Second output",
        agent=researcher,
        condition=condition_fails,
    )

    conditional_task2 = ConditionalTask(
        description="Third task - conditional that succeeds using first task output",
        expected_output="Third output",
        agent=writer,
        condition=condition_succeeds,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, conditional_task1, conditional_task2],
    )

    # Mock outputs for tasks
    mock_first = TaskOutput(
        description="First task output",
        raw="First success output",  # Will be used by third task's condition
        agent=researcher.role,
        messages=[],
    )
    mock_third = TaskOutput(
        description="Third task output",
        raw="Third task executed",  # Output when condition succeeds using first task output
        agent=writer.role,
        messages=[],
    )

    # Set up mocks for task execution and conditional logic
    with patch.object(ConditionalTask, "should_execute") as mock_should_execute:
        # First conditional fails, second succeeds
        mock_should_execute.side_effect = [False, True]
        with patch.object(Task, "execute_sync") as mock_execute:
            mock_execute.side_effect = [mock_first, mock_third]
            result = crew.kickoff()

            # Verify execution behavior
            assert mock_execute.call_count == 2  # Only first and third tasks execute
            assert mock_should_execute.call_count == 2  # Both conditionals checked

            # Verify outputs collection:
            # First executed task output, followed by an automatically generated (skipped) output, then the conditional execution
            assert len(result.tasks_output) == 3
            assert (
                result.tasks_output[0].raw == "First success output"
            )  # First task succeeded
            assert (
                result.tasks_output[1].raw == ""
            )  # Second task skipped (condition failed)
            assert (
                result.tasks_output[2].raw == "Third task executed"
            )  # Third task used first task's output


@pytest.mark.vcr()
def test_conditional_tasks_result_collection(researcher, writer):
    """Test that task outputs are properly collected based on execution status."""
    task1 = Task(
        description="Normal task that always executes",
        expected_output="First output",
        agent=researcher,
    )

    def condition_never_met(task_output: TaskOutput) -> bool:
        return "never matches" in task_output.raw.lower()

    def condition_always_met(task_output: TaskOutput) -> bool:
        return "success" in task_output.raw.lower()

    task2 = ConditionalTask(
        description="Conditional task that never executes",
        expected_output="Second output",
        agent=researcher,
        condition=condition_never_met,
    )

    task3 = ConditionalTask(
        description="Conditional task that always executes",
        expected_output="Third output",
        agent=writer,
        condition=condition_always_met,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2, task3],
    )

    # Mock outputs for different execution paths
    mock_success = TaskOutput(
        description="Success output",
        raw="Success output",  # Triggers third task's condition
        agent=researcher.role,
        messages=[],
    )
    mock_conditional = TaskOutput(
        description="Conditional output",
        raw="Conditional task executed",
        agent=writer.role,
        messages=[],
    )

    # Set up mocks for task execution and conditional logic
    with patch.object(ConditionalTask, "should_execute") as mock_should_execute:
        # First conditional fails, second succeeds
        mock_should_execute.side_effect = [False, True]
        with patch.object(Task, "execute_sync") as mock_execute:
            mock_execute.side_effect = [mock_success, mock_conditional]
            result = crew.kickoff()

            # Verify execution behavior
            assert mock_execute.call_count == 2  # Only first and third tasks execute
            assert mock_should_execute.call_count == 2  # Both conditionals checked

            # Verify task output collection:
            # There should be three outputs: normal task, skipped conditional task (empty output),
            # and the conditional task that executed.
            assert len(result.tasks_output) == 3
            assert (
                result.tasks_output[0].raw == "Success output"
            )  # Normal task executed
            assert result.tasks_output[1].raw == ""  # Second task skipped
            assert (
                result.tasks_output[2].raw == "Conditional task executed"
            )  # Third task executed

            # Verify task output collection
            assert len(result.tasks_output) == 3
            assert (
                result.tasks_output[0].raw == "Success output"
            )  # Normal task executed
            assert result.tasks_output[1].raw == ""  # Second task skipped
            assert (
                result.tasks_output[2].raw == "Conditional task executed"
            )  # Third task executed


@pytest.mark.vcr()
def test_multiple_conditional_tasks(researcher, writer):
    """Test that having multiple conditional tasks in sequence works correctly."""
    task1 = Task(
        description="Initial research task",
        expected_output="Research output",
        agent=researcher,
    )

    def condition1(task_output: TaskOutput) -> bool:
        return "success" in task_output.raw.lower()

    def condition2(task_output: TaskOutput) -> bool:
        return "proceed" in task_output.raw.lower()

    task2 = ConditionalTask(
        description="First conditional task",
        expected_output="Conditional output 1",
        agent=writer,
        condition=condition1,
    )

    task3 = ConditionalTask(
        description="Second conditional task",
        expected_output="Conditional output 2",
        agent=writer,
        condition=condition2,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2, task3],
    )

    # Mock different task outputs to test conditional logic
    mock_success = TaskOutput(
        description="Mock success",
        raw="Success and proceed output",
        agent=researcher.role,
        messages=[],
    )

    # Set up mocks for task execution
    with patch.object(Task, "execute_sync", return_value=mock_success) as mock_execute:
        result = crew.kickoff()
        # Verify all tasks were executed (no IndexError)
        assert mock_execute.call_count == 3
        assert len(result.tasks_output) == 3


@pytest.mark.vcr()
def test_using_contextual_memory():
    math_researcher = Agent(
        role="Researcher",
        goal="You research about math.",
        backstory="You're an expert in research and you love to learn new things.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Research a topic to teach a kid aged 6 about math.",
        expected_output="A topic, explanation, angle, and examples.",
        agent=math_researcher,
    )

    crew = Crew(
        agents=[math_researcher],
        tasks=[task1],
        memory=True,
    )

    with patch.object(
        ContextualMemory, "build_context_for_task", return_value=""
    ) as contextual_mem:
        crew.kickoff()
        contextual_mem.assert_called_once()


@pytest.mark.vcr()
def test_memory_events_are_emitted():
    events = defaultdict(list)
    condition = threading.Condition()

    @crewai_event_bus.on(MemorySaveStartedEvent)
    def handle_memory_save_started(source, event):
        with condition:
            events["MemorySaveStartedEvent"].append(event)
            condition.notify_all()

    @crewai_event_bus.on(MemorySaveCompletedEvent)
    def handle_memory_save_completed(source, event):
        with condition:
            events["MemorySaveCompletedEvent"].append(event)
            condition.notify_all()

    @crewai_event_bus.on(MemorySaveFailedEvent)
    def handle_memory_save_failed(source, event):
        with condition:
            events["MemorySaveFailedEvent"].append(event)
            condition.notify_all()

    @crewai_event_bus.on(MemoryQueryStartedEvent)
    def handle_memory_query_started(source, event):
        with condition:
            events["MemoryQueryStartedEvent"].append(event)
            condition.notify_all()

    @crewai_event_bus.on(MemoryQueryCompletedEvent)
    def handle_memory_query_completed(source, event):
        with condition:
            events["MemoryQueryCompletedEvent"].append(event)
            condition.notify_all()

    @crewai_event_bus.on(MemoryQueryFailedEvent)
    def handle_memory_query_failed(source, event):
        with condition:
            events["MemoryQueryFailedEvent"].append(event)
            condition.notify_all()

    @crewai_event_bus.on(MemoryRetrievalStartedEvent)
    def handle_memory_retrieval_started(source, event):
        with condition:
            events["MemoryRetrievalStartedEvent"].append(event)
            condition.notify_all()

    @crewai_event_bus.on(MemoryRetrievalCompletedEvent)
    def handle_memory_retrieval_completed(source, event):
        with condition:
            events["MemoryRetrievalCompletedEvent"].append(event)
            condition.notify_all()

    math_researcher = Agent(
        role="Researcher",
        goal="You research about math.",
        backstory="You're an expert in research and you love to learn new things.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Research a topic to teach a kid aged 6 about math.",
        expected_output="A topic, explanation, angle, and examples.",
        agent=math_researcher,
    )

    crew = Crew(
        agents=[math_researcher],
        tasks=[task1],
        memory=True,
    )

    crew.kickoff()

    with condition:
        success = condition.wait_for(
            lambda: (
                len(events["MemorySaveStartedEvent"]) >= 3
                and len(events["MemorySaveCompletedEvent"]) >= 3
                and len(events["MemoryQueryStartedEvent"]) >= 3
                and len(events["MemoryQueryCompletedEvent"]) >= 3
                and len(events["MemoryRetrievalCompletedEvent"]) >= 1
            ),
            timeout=10,
        )

    assert success, f"Timeout waiting for memory events. Got: {dict(events)}"
    assert len(events["MemorySaveStartedEvent"]) == 3
    assert len(events["MemorySaveCompletedEvent"]) == 3
    assert len(events["MemorySaveFailedEvent"]) == 0
    assert len(events["MemoryQueryStartedEvent"]) == 3
    assert len(events["MemoryQueryCompletedEvent"]) == 3
    assert len(events["MemoryQueryFailedEvent"]) == 0
    assert len(events["MemoryRetrievalStartedEvent"]) == 1
    assert len(events["MemoryRetrievalCompletedEvent"]) == 1


@pytest.mark.vcr()
def test_using_contextual_memory_with_long_term_memory():
    math_researcher = Agent(
        role="Researcher",
        goal="You research about math.",
        backstory="You're an expert in research and you love to learn new things.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Research a topic to teach a kid aged 6 about math.",
        expected_output="A topic, explanation, angle, and examples.",
        agent=math_researcher,
    )

    crew = Crew(
        agents=[math_researcher],
        tasks=[task1],
        long_term_memory=LongTermMemory(),
    )

    with patch.object(
        ContextualMemory, "build_context_for_task", return_value=""
    ) as contextual_mem:
        crew.kickoff()
        contextual_mem.assert_called_once()
        assert crew.memory is False


@pytest.mark.vcr()
def test_warning_long_term_memory_without_entity_memory():
    math_researcher = Agent(
        role="Researcher",
        goal="You research about math.",
        backstory="You're an expert in research and you love to learn new things.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Research a topic to teach a kid aged 6 about math.",
        expected_output="A topic, explanation, angle, and examples.",
        agent=math_researcher,
    )

    crew = Crew(
        agents=[math_researcher],
        tasks=[task1],
        long_term_memory=LongTermMemory(),
    )

    with (
        patch("crewai.utilities.printer.Printer.print") as mock_print,
        patch(
            "crewai.memory.long_term.long_term_memory.LongTermMemory.save"
        ) as save_memory,
    ):
        crew.kickoff()
        mock_print.assert_called_with(
            content="Long term memory is enabled, but entity memory is not enabled. Please configure entity memory or set memory=True to automatically enable it.",
            color="bold_yellow",
        )
        save_memory.assert_not_called()


@pytest.mark.vcr()
def test_long_term_memory_with_memory_flag():
    math_researcher = Agent(
        role="Researcher",
        goal="You research about math.",
        backstory="You're an expert in research and you love to learn new things.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Research a topic to teach a kid aged 6 about math.",
        expected_output="A topic, explanation, angle, and examples.",
        agent=math_researcher,
    )

    with (
        patch("crewai.utilities.printer.Printer.print") as mock_print,
        patch("crewai.memory.long_term.long_term_memory.LongTermMemory.save") as save_memory,
    ):
        crew = Crew(
            agents=[math_researcher],
            tasks=[task1],
            memory=True,
            long_term_memory=LongTermMemory(),
        )
        crew.kickoff()
        mock_print.assert_not_called()
        save_memory.assert_called_once()


@pytest.mark.vcr()
def test_using_contextual_memory_with_short_term_memory():
    math_researcher = Agent(
        role="Researcher",
        goal="You research about math.",
        backstory="You're an expert in research and you love to learn new things.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Research a topic to teach a kid aged 6 about math.",
        expected_output="A topic, explanation, angle, and examples.",
        agent=math_researcher,
    )

    crew = Crew(
        agents=[math_researcher],
        tasks=[task1],
        short_term_memory=ShortTermMemory(),
    )

    with patch.object(
        ContextualMemory, "build_context_for_task", return_value=""
    ) as contextual_mem:
        crew.kickoff()
        contextual_mem.assert_called_once()
        assert crew.memory is False


@pytest.mark.vcr()
def test_disabled_memory_using_contextual_memory():
    math_researcher = Agent(
        role="Researcher",
        goal="You research about math.",
        backstory="You're an expert in research and you love to learn new things.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Research a topic to teach a kid aged 6 about math.",
        expected_output="A topic, explanation, angle, and examples.",
        agent=math_researcher,
    )

    crew = Crew(
        agents=[math_researcher],
        tasks=[task1],
        memory=False,
    )

    with patch.object(
        ContextualMemory, "build_context_for_task", return_value=""
    ) as contextual_mem:
        crew.kickoff()
        contextual_mem.assert_not_called()


@pytest.mark.vcr()
def test_crew_log_file_output(tmp_path, researcher):
    test_file = tmp_path / "logs.txt"
    tasks = [
        Task(
            description="Say Hi",
            expected_output="The word: Hi",
            agent=researcher,
        )
    ]

    crew = Crew(agents=[researcher], tasks=tasks, output_log_file=str(test_file))
    crew.kickoff()
    assert test_file.exists()


@pytest.mark.vcr()
def test_crew_output_file_end_to_end(tmp_path):
    """Test output file functionality in a full crew context."""
    # Create an agent
    agent = Agent(
        role="Researcher",
        goal="Analyze AI topics",
        backstory="You have extensive AI research experience.",
        allow_delegation=False,
    )

    # Create a task with dynamic output file path
    dynamic_path = tmp_path / "output_{topic}.txt"
    task = Task(
        description="Explain the advantages of {topic}.",
        expected_output="A summary of the main advantages, bullet points recommended.",
        agent=agent,
        output_file=str(dynamic_path),
    )

    # Create and run the crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
    )
    crew.kickoff(inputs={"topic": "AI"})

    # Verify file creation and cleanup
    expected_file = tmp_path / "output_AI.txt"
    assert expected_file.exists(), f"Output file {expected_file} was not created"


def test_crew_output_file_validation_failures():
    """Test output file validation failures in a crew context."""
    agent = Agent(
        role="Researcher",
        goal="Analyze data",
        backstory="You analyze data files.",
        allow_delegation=False,
    )

    # Test path traversal
    with pytest.raises(ValueError, match="Path traversal"):
        task = Task(
            description="Analyze data",
            expected_output="Analysis results",
            agent=agent,
            output_file="../output.txt",
        )
        Crew(agents=[agent], tasks=[task]).kickoff()

    # Test shell special characters
    with pytest.raises(ValueError, match="Shell special characters"):
        task = Task(
            description="Analyze data",
            expected_output="Analysis results",
            agent=agent,
            output_file="output.txt | rm -rf /",
        )
        Crew(agents=[agent], tasks=[task]).kickoff()

    # Test shell expansion
    with pytest.raises(ValueError, match="Shell expansion"):
        task = Task(
            description="Analyze data",
            expected_output="Analysis results",
            agent=agent,
            output_file="~/output.txt",
        )
        Crew(agents=[agent], tasks=[task]).kickoff()

    # Test invalid template variable
    with pytest.raises(ValueError, match="Invalid template variable"):
        task = Task(
            description="Analyze data",
            expected_output="Analysis results",
            agent=agent,
            output_file="{invalid-name}/output.txt",
        )
        Crew(agents=[agent], tasks=[task]).kickoff()


@pytest.mark.vcr()
def test_manager_agent(researcher, writer):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
    )

    manager = Agent(
        role="Manager",
        goal="Manage the crew and ensure the tasks are completed efficiently.",
        backstory="You're an experienced manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
        allow_delegation=False,
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.hierarchical,
        manager_agent=manager,
        tasks=[task],
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Because we are mocking execute_sync, we never hit the underlying _execute_core
    # which sets the output attribute of the task
    task.output = mock_task_output

    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()
        assert manager.allow_delegation is True
        mock_execute_sync.assert_called()


def test_manager_agent_in_agents_raises_exception(researcher, writer):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
    )

    manager = Agent(
        role="Manager",
        goal="Manage the crew and ensure the tasks are completed efficiently.",
        backstory="You're an experienced manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
        allow_delegation=False,
    )

    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        Crew(
            agents=[researcher, writer, manager],
            process=Process.hierarchical,
            manager_agent=manager,
            tasks=[task],
        )


@pytest.mark.vcr()
def test_manager_agent_with_tools_raises_exception(researcher, writer):
    @tool
    def testing_tool(first_number: int, second_number: int) -> int:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
    )

    manager = Agent(
        role="Manager",
        goal="Manage the crew and ensure the tasks are completed efficiently.",
        backstory="You're an experienced manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
        allow_delegation=False,
        tools=[testing_tool],
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.hierarchical,
        manager_agent=manager,
        tasks=[task],
    )

    with pytest.raises(Exception, match="Manager agent should not have tools"):
        crew.kickoff()


@pytest.mark.vcr()
def test_crew_train_success(researcher, writer, monkeypatch):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
    )

    received_events = []
    condition = threading.Condition()

    @crewai_event_bus.on(CrewTrainStartedEvent)
    def on_crew_train_started(source, event: CrewTrainStartedEvent):
        with condition:
            received_events.append(event)
            condition.notify()

    @crewai_event_bus.on(CrewTrainCompletedEvent)
    def on_crew_train_completed(source, event: CrewTrainCompletedEvent):
        with condition:
            received_events.append(event)
            condition.notify()

    # Mock human input to avoid blocking during training
    # Use StringIO to simulate user input for multiple calls to input()
    mock_inputs = StringIO("Great work!\n" * 10)  # Provide enough inputs for all iterations
    monkeypatch.setattr("sys.stdin", mock_inputs)

    crew.train(
        n_iterations=2, inputs={"topic": "AI"}, filename="trained_agents_data.pkl"
    )

    with condition:
        success = condition.wait_for(lambda: len(received_events) == 2, timeout=5)

    assert success, "Timeout waiting for all train events"
    assert len(received_events) == 2
    assert isinstance(received_events[0], CrewTrainStartedEvent)
    assert isinstance(received_events[1], CrewTrainCompletedEvent)


@pytest.mark.vcr()
def test_crew_train_error(researcher, writer):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article",
        expected_output="5 bullet points with a paragraph for each idea.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
    )

    with pytest.raises(TypeError) as e:
        crew.train()  # type: ignore purposefully throwing err
        assert "train() missing 1 required positional argument: 'n_iterations'" in str(
            e
        )


def test__setup_for_training(researcher, writer):
    researcher.allow_delegation = True
    writer.allow_delegation = True
    agents = [researcher, writer]
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article",
        expected_output="5 bullet points with a paragraph for each idea.",
        agent=researcher,
    )

    crew = Crew(
        agents=agents,
        tasks=[task],
    )

    assert crew._train is False
    assert task.human_input is False

    for agent in agents:
        assert agent.allow_delegation is True

    crew._setup_for_training("trained_agents_data.pkl")

    assert crew._train is True
    assert task.human_input is True

    for agent in agents:
        assert agent.allow_delegation is False


@pytest.mark.vcr()
def test_replay_feature(researcher, writer):
    list_ideas = Task(
        description="Generate a list of 5 interesting ideas to explore for an article, where each bulletpoint is under 15 words.",
        expected_output="Bullet point list of 5 important events. No additional commentary.",
        agent=researcher,
    )
    write = Task(
        description="Write a sentence about the events",
        expected_output="A sentence about the events",
        agent=writer,
        context=[list_ideas],
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[list_ideas, write],
        process=Process.sequential,
    )

    with patch.object(Task, "execute_sync") as mock_execute_task:
        mock_execute_task.return_value = TaskOutput(
            description="Mock description",
            raw="Mocked output for list of ideas",
            agent="Researcher",
            json_dict=None,
            output_format=OutputFormat.RAW,
            pydantic=None,
            summary="Mocked output for list of ideas",
            messages=[],
        )

        crew.kickoff()
        crew.replay(str(write.id))
        # Ensure context was passed correctly
        assert mock_execute_task.call_count == 3


@pytest.mark.vcr()
def test_crew_replay_error(researcher, writer):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article",
        expected_output="5 bullet points with a paragraph for each idea.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
    )

    with pytest.raises(TypeError) as e:
        crew.replay()  # type: ignore purposefully throwing err
        assert "task_id is required" in str(e)


@pytest.mark.vcr()
def test_crew_task_db_init():
    agent = Agent(
        role="Content Writer",
        goal="Write engaging content on various topics.",
        backstory="You have a background in journalism and creative writing.",
    )

    task = Task(
        description="Write a detailed article about AI in healthcare.",
        expected_output="A 1 paragraph article about AI.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])

    with patch.object(Task, "execute_sync") as mock_execute_task:
        mock_execute_task.return_value = TaskOutput(
            description="Write about AI in healthcare.",
            raw="Artificial Intelligence (AI) is revolutionizing healthcare by enhancing diagnostic accuracy, personalizing treatment plans, and streamlining administrative tasks.",
            agent="Content Writer",
            json_dict=None,
            output_format=OutputFormat.RAW,
            pydantic=None,
            summary="Write about AI in healthcare...",
            messages=[],
        )

        crew.kickoff()

        # Check if this runs without raising an exception
        try:
            db_handler = TaskOutputStorageHandler()
            db_handler.load()
            assert True  # If we reach this point, no exception was raised
        except Exception as e:
            pytest.fail(f"An exception was raised: {e!s}")


@pytest.mark.vcr()
def test_replay_task_with_context():
    agent1 = Agent(
        role="Researcher",
        goal="Research AI advancements.",
        backstory="You are an expert in AI research.",
    )
    agent2 = Agent(
        role="Writer",
        goal="Write detailed articles on AI.",
        backstory="You have a background in journalism and AI.",
    )

    task1 = Task(
        description="Research the latest advancements in AI.",
        expected_output="A detailed report on AI advancements.",
        agent=agent1,
    )
    task2 = Task(
        description="Summarize the AI advancements report.",
        expected_output="A summary of the AI advancements report.",
        agent=agent2,
    )
    task3 = Task(
        description="Write an article based on the AI advancements summary.",
        expected_output="An article on AI advancements.",
        agent=agent2,
    )
    task4 = Task(
        description="Create a presentation based on the AI advancements article.",
        expected_output="A presentation on AI advancements.",
        agent=agent2,
        context=[task1],
    )

    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
    )

    mock_task_output1 = TaskOutput(
        description="Research the latest advancements in AI.",
        raw="Detailed report on AI advancements...",
        agent="Researcher",
        json_dict=None,
        output_format=OutputFormat.RAW,
        pydantic=None,
        summary="Detailed report on AI advancements...",
        messages=[],
    )
    mock_task_output2 = TaskOutput(
        description="Summarize the AI advancements report.",
        raw="Summary of the AI advancements report...",
        agent="Writer",
        json_dict=None,
        output_format=OutputFormat.RAW,
        pydantic=None,
        summary="Summary of the AI advancements report...",
        messages=[],
    )
    mock_task_output3 = TaskOutput(
        description="Write an article based on the AI advancements summary.",
        raw="Article on AI advancements...",
        agent="Writer",
        json_dict=None,
        output_format=OutputFormat.RAW,
        pydantic=None,
        summary="Article on AI advancements...",
        messages=[],
    )
    mock_task_output4 = TaskOutput(
        description="Create a presentation based on the AI advancements article.",
        raw="Presentation on AI advancements...",
        agent="Writer",
        json_dict=None,
        output_format=OutputFormat.RAW,
        pydantic=None,
        summary="Presentation on AI advancements...",
        messages=[],
    )

    with patch.object(Task, "execute_sync") as mock_execute_task:
        mock_execute_task.side_effect = [
            mock_task_output1,
            mock_task_output2,
            mock_task_output3,
            mock_task_output4,
        ]

        crew.kickoff()
        db_handler = TaskOutputStorageHandler()
        assert db_handler.load() != []

        with patch.object(Task, "execute_sync") as mock_replay_task:
            mock_replay_task.return_value = mock_task_output4

            replayed_output = crew.replay(str(task4.id))
            assert replayed_output.raw == "Presentation on AI advancements..."

        db_handler.reset()


@pytest.mark.vcr()
def test_replay_preserves_messages():
    """Test that replay preserves messages from stored task outputs."""
    from crewai.utilities.types import LLMMessage

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        allow_delegation=False,
    )

    task = Task(
        description="Say hello",
        expected_output="A greeting",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

    mock_messages: list[LLMMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello!"},
    ]

    mock_task_output = TaskOutput(
        description="Say hello",
        raw="Hello!",
        agent="Test Agent",
        messages=mock_messages,
    )

    with patch.object(Task, "execute_sync", return_value=mock_task_output):
        crew.kickoff()

    # Verify the task output was stored with messages
    db_handler = TaskOutputStorageHandler()
    stored_outputs = db_handler.load()
    assert stored_outputs is not None
    assert len(stored_outputs) > 0

    # Verify messages are in the stored output
    stored_output = stored_outputs[0]["output"]
    assert "messages" in stored_output
    assert len(stored_output["messages"]) == 3
    assert stored_output["messages"][0]["role"] == "system"
    assert stored_output["messages"][1]["role"] == "user"
    assert stored_output["messages"][2]["role"] == "assistant"

    # Replay the task and verify messages are preserved
    with patch.object(Task, "execute_sync", return_value=mock_task_output):
        replayed_output = crew.replay(str(task.id))

    # Verify the replayed task output has messages
    assert len(replayed_output.tasks_output) > 0
    replayed_task_output = replayed_output.tasks_output[0]
    assert hasattr(replayed_task_output, "messages")
    assert isinstance(replayed_task_output.messages, list)
    assert len(replayed_task_output.messages) == 3

    db_handler.reset()


@pytest.mark.vcr()
def test_replay_with_context():
    agent = Agent(role="test_agent", backstory="Test Description", goal="Test Goal")
    task1 = Task(
        description="Context Task", expected_output="Say Task Output", agent=agent
    )
    task2 = Task(
        description="Test Task", expected_output="Say Hi", agent=agent, context=[task1]
    )

    context_output = TaskOutput(
        description="Context Task Output",
        agent="test_agent",
        raw="context raw output",
        pydantic=None,
        json_dict={},
        output_format=OutputFormat.RAW,
        messages=[],
    )
    task1.output = context_output

    crew = Crew(agents=[agent], tasks=[task1, task2], process=Process.sequential)

    with patch(
        "crewai.utilities.task_output_storage_handler.TaskOutputStorageHandler.load",
        return_value=[
            {
                "task_id": str(task1.id),
                "output": {
                    "description": context_output.description,
                    "summary": context_output.summary,
                    "raw": context_output.raw,
                    "pydantic": context_output.pydantic,
                    "json_dict": context_output.json_dict,
                    "output_format": context_output.output_format,
                    "agent": context_output.agent,
                },
                "inputs": {},
            },
            {
                "task_id": str(task2.id),
                "output": {
                    "description": "Test Task Output",
                    "summary": None,
                    "raw": "test raw output",
                    "pydantic": None,
                    "json_dict": {},
                    "output_format": "json",
                    "agent": "test_agent",
                },
                "inputs": {},
            },
        ],
    ):
        crew.replay(str(task2.id))

        assert crew.tasks[1].context[0].output.raw == "context raw output"


@pytest.mark.vcr()
def test_replay_with_context_set_to_nullable():
    agent = Agent(role="test_agent", backstory="Test Description", goal="Test Goal")
    task1 = Task(
        description="Context Task", expected_output="Say Task Output", agent=agent
    )
    task2 = Task(
        description="Test Task", expected_output="Say Hi", agent=agent, context=[]
    )
    task3 = Task(
        description="Test Task 3", expected_output="Say Hi", agent=agent, context=None
    )

    crew = Crew(agents=[agent], tasks=[task1, task2, task3], process=Process.sequential)
    with patch("crewai.task.Task.execute_sync") as mock_execute_task:
        mock_execute_task.return_value = TaskOutput(
            description="Test Task Output",
            raw="test raw output",
            agent="test_agent",
            messages=[],
        )
        crew.kickoff()

    mock_execute_task.assert_called_with(agent=ANY, context="", tools=ANY)


@pytest.mark.vcr()
def test_replay_with_invalid_task_id():
    agent = Agent(role="test_agent", backstory="Test Description", goal="Test Goal")
    task1 = Task(
        description="Context Task", expected_output="Say Task Output", agent=agent
    )
    task2 = Task(
        description="Test Task", expected_output="Say Hi", agent=agent, context=[task1]
    )

    context_output = TaskOutput(
        description="Context Task Output",
        agent="test_agent",
        raw="context raw output",
        pydantic=None,
        json_dict={},
        output_format=OutputFormat.RAW,
        messages=[],
    )
    task1.output = context_output

    crew = Crew(agents=[agent], tasks=[task1, task2], process=Process.sequential)

    with patch(
        "crewai.utilities.task_output_storage_handler.TaskOutputStorageHandler.load",
        return_value=[
            {
                "task_id": str(task1.id),
                "output": {
                    "description": context_output.description,
                    "summary": context_output.summary,
                    "raw": context_output.raw,
                    "pydantic": context_output.pydantic,
                    "json_dict": context_output.json_dict,
                    "output_format": context_output.output_format,
                    "agent": context_output.agent,
                },
                "inputs": {},
            },
            {
                "task_id": str(task2.id),
                "output": {
                    "description": "Test Task Output",
                    "summary": None,
                    "raw": "test raw output",
                    "pydantic": None,
                    "json_dict": {},
                    "output_format": "json",
                    "agent": "test_agent",
                },
                "inputs": {},
            },
        ],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Task with id bf5b09c9-69bd-4eb8-be12-f9e5bae31c2d not found in the crew's tasks."
            ),
        ):
            crew.replay("bf5b09c9-69bd-4eb8-be12-f9e5bae31c2d")


@pytest.mark.vcr()
@patch.object(Crew, "_interpolate_inputs")
def test_replay_interpolates_inputs_properly(mock_interpolate_inputs):
    agent = Agent(role="test_agent", backstory="Test Description", goal="Test Goal")
    task1 = Task(description="Context Task", expected_output="Say {name}", agent=agent)
    task2 = Task(
        description="Test Task",
        expected_output="Say Hi to {name}",
        agent=agent,
        context=[task1],
    )

    context_output = TaskOutput(
        description="Context Task Output",
        agent="test_agent",
        raw="context raw output",
        pydantic=None,
        json_dict={},
        output_format=OutputFormat.RAW,
        messages=[],
    )
    task1.output = context_output

    crew = Crew(agents=[agent], tasks=[task1, task2], process=Process.sequential)
    crew.kickoff(inputs={"name": "John"})

    with patch(
        "crewai.utilities.task_output_storage_handler.TaskOutputStorageHandler.load",
        return_value=[
            {
                "task_id": str(task1.id),
                "output": {
                    "description": context_output.description,
                    "summary": context_output.summary,
                    "raw": context_output.raw,
                    "pydantic": context_output.pydantic,
                    "json_dict": context_output.json_dict,
                    "output_format": context_output.output_format,
                    "agent": context_output.agent,
                },
                "inputs": {"name": "John"},
            },
            {
                "task_id": str(task2.id),
                "output": {
                    "description": "Test Task Output",
                    "summary": None,
                    "raw": "test raw output",
                    "pydantic": None,
                    "json_dict": {},
                    "output_format": "json",
                    "agent": "test_agent",
                },
                "inputs": {"name": "John"},
            },
        ],
    ):
        crew.replay(str(task2.id))
        assert crew._inputs == {"name": "John"}
        assert mock_interpolate_inputs.call_count == 2


@pytest.mark.vcr()
def test_replay_setup_context():
    agent = Agent(role="test_agent", backstory="Test Description", goal="Test Goal")
    task1 = Task(description="Context Task", expected_output="Say {name}", agent=agent)
    task2 = Task(
        description="Test Task",
        expected_output="Say Hi to {name}",
        agent=agent,
    )
    context_output = TaskOutput(
        description="Context Task Output",
        agent="test_agent",
        raw="context raw output",
        pydantic=None,
        json_dict={},
        output_format=OutputFormat.RAW,
        messages=[],
    )
    task1.output = context_output
    crew = Crew(agents=[agent], tasks=[task1, task2], process=Process.sequential)
    with patch(
        "crewai.utilities.task_output_storage_handler.TaskOutputStorageHandler.load",
        return_value=[
            {
                "task_id": str(task1.id),
                "output": {
                    "description": context_output.description,
                    "summary": context_output.summary,
                    "raw": context_output.raw,
                    "pydantic": context_output.pydantic,
                    "json_dict": context_output.json_dict,
                    "output_format": context_output.output_format,
                    "agent": context_output.agent,
                },
                "inputs": {"name": "John"},
            },
            {
                "task_id": str(task2.id),
                "output": {
                    "description": "Test Task Output",
                    "summary": None,
                    "raw": "test raw output",
                    "pydantic": None,
                    "json_dict": {},
                    "output_format": "json",
                    "agent": "test_agent",
                },
                "inputs": {"name": "John"},
            },
        ],
    ):
        crew.replay(str(task2.id))

        # Check if the first task's output was set correctly
        assert crew.tasks[0].output is not None
        assert isinstance(crew.tasks[0].output, TaskOutput)
        assert crew.tasks[0].output.description == "Context Task Output"
        assert crew.tasks[0].output.agent == "test_agent"
        assert crew.tasks[0].output.raw == "context raw output"
        assert crew.tasks[0].output.output_format == OutputFormat.RAW

        assert crew.tasks[1].prompt_context == "context raw output"


def test_key(researcher, writer):
    tasks = [
        Task(
            description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
            expected_output="Bullet point list of 5 important events.",
            agent=researcher,
        ),
        Task(
            description="Write a 1 amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
            expected_output="A 4 paragraph article about AI.",
            agent=writer,
        ),
    ]
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=tasks,
    )
    hash = md5(
        f"{researcher.key}|{writer.key}|{tasks[0].key}|{tasks[1].key}".encode(),
        usedforsecurity=False,
    ).hexdigest()

    assert crew.key == hash


def test_key_with_interpolated_inputs():
    researcher = Agent(
        role="{topic} Researcher",
        goal="Make the best research and analysis on content {topic}",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    writer = Agent(
        role="{topic} Senior Writer",
        goal="Write the best content about {topic}",
        backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
        allow_delegation=False,
    )

    tasks = [
        Task(
            description="Give me a list of 5 interesting ideas about {topic} to explore for an article, what makes them unique and interesting.",
            expected_output="Bullet point list of 5 important events.",
            agent=researcher,
        ),
        Task(
            description="Write a 1 amazing paragraph highlight for each idea of {topic} that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
            expected_output="A 4 paragraph article about AI.",
            agent=writer,
        ),
    ]

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=tasks,
    )
    hash = md5(
        f"{researcher.key}|{writer.key}|{tasks[0].key}|{tasks[1].key}".encode(),
        usedforsecurity=False,
    ).hexdigest()

    assert crew.key == hash

    curr_key = crew.key
    crew._interpolate_inputs({"topic": "AI"})
    assert crew.key == curr_key


def test_conditional_task_requirement_breaks_when_singular_conditional_task(
    researcher, writer
):
    def condition_fn(output) -> bool:
        return output.raw.startswith("Andrew Ng has!!")

    task = ConditionalTask(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        condition=condition_fn,
    )

    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        Crew(
            agents=[researcher, writer],
            tasks=[task],
        )


@pytest.mark.vcr()
def test_conditional_task_last_task_when_conditional_is_true(researcher, writer):
    def condition_fn(output) -> bool:
        return True

    task1 = Task(
        description="Say Hi",
        expected_output="Hi",
        agent=researcher,
    )
    task2 = ConditionalTask(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        condition=condition_fn,
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
    )
    result = crew.kickoff()
    assert result.raw.startswith(
        "Hi\n\nHere are five interesting ideas for articles focused on AI and AI agents, each accompanied by a compelling paragraph to showcase the potential impact and depth of each topic:"
    )


@pytest.mark.vcr()
def test_conditional_task_last_task_when_conditional_is_false(researcher, writer):
    def condition_fn(output) -> bool:
        return False

    task1 = Task(
        description="Say Hi",
        expected_output="Hi",
        agent=researcher,
    )
    task2 = ConditionalTask(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        condition=condition_fn,
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
    )
    result = crew.kickoff()
    assert result.raw == "Hi"


def test_conditional_task_requirement_breaks_when_task_async(researcher, writer):
    def my_condition(context):
        return context.get("some_value") > 10

    task = ConditionalTask(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        execute_async=True,
        condition=my_condition,
        agent=researcher,
    )
    task2 = Task(
        description="Say Hi",
        expected_output="Hi",
        agent=writer,
    )

    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        Crew(
            agents=[researcher, writer],
            tasks=[task, task2],
        )


@pytest.mark.vcr()
def test_conditional_should_skip(researcher, writer):
    task1 = Task(description="Return hello", expected_output="say hi", agent=researcher)

    condition_mock = MagicMock(return_value=False)
    task2 = ConditionalTask(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        condition=condition_mock,
        agent=writer,
    )
    crew_met = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
    )
    with patch.object(Task, "execute_sync") as mock_execute_sync:
        mock_execute_sync.return_value = TaskOutput(
            description="Task 1 description",
            raw="Task 1 output",
            agent="Researcher",
            messages=[],
        )

        result = crew_met.kickoff()
        assert mock_execute_sync.call_count == 1

        assert condition_mock.call_count == 1
        assert condition_mock() is False

        assert task2.output is None
        assert result.raw.startswith("Task 1 output")


@pytest.mark.vcr()
def test_conditional_should_execute(researcher, writer):
    task1 = Task(description="Return hello", expected_output="say hi", agent=researcher)

    condition_mock = MagicMock(
        return_value=True
    )  # should execute this conditional task
    task2 = ConditionalTask(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        condition=condition_mock,
        agent=writer,
    )
    crew_met = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
    )
    with patch.object(Task, "execute_sync") as mock_execute_sync:
        mock_execute_sync.return_value = TaskOutput(
            description="Task 1 description",
            raw="Task 1 output",
            agent="Researcher",
            messages=[],
        )

        crew_met.kickoff()

        assert condition_mock.call_count == 1
        assert condition_mock() is True
        assert mock_execute_sync.call_count == 2


@pytest.mark.vcr()
def test_crew_testing_function(researcher):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
    )

    n_iterations = 2
    llm_instance = LLM("gpt-4o-mini")

    received_events = []
    condition = threading.Condition()

    @crewai_event_bus.on(CrewTestStartedEvent)
    def on_crew_test_started(source, event: CrewTestStartedEvent):
        with condition:
            received_events.append(event)
            condition.notify()

    @crewai_event_bus.on(CrewTestCompletedEvent)
    def on_crew_test_completed(source, event: CrewTestCompletedEvent):
        with condition:
            received_events.append(event)
            condition.notify()

    crew.test(n_iterations, llm_instance, inputs={"topic": "AI"})

    with condition:
        success = condition.wait_for(lambda: len(received_events) == 2, timeout=5)

    assert success, "Timeout waiting for all test events"
    assert len(received_events) == 2
    assert isinstance(received_events[0], CrewTestStartedEvent)
    assert isinstance(received_events[1], CrewTestCompletedEvent)


@pytest.mark.vcr()
def test_hierarchical_verbose_manager_agent(researcher, writer):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        verbose=True,
    )

    crew.kickoff()

    assert crew.manager_agent is not None
    assert crew.manager_agent.verbose


@pytest.mark.vcr()
def test_hierarchical_verbose_false_manager_agent(researcher, writer):
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="5 bullet points with a paragraph for each idea.",
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        verbose=False,
    )

    crew.kickoff()

    assert crew.manager_agent is not None
    assert not crew.manager_agent.verbose


def test_fetch_inputs():
    agent = Agent(
        role="{role_detail} Researcher",
        goal="Research on {topic}.",
        backstory="Expert in {field}.",
    )

    task = Task(
        description="Analyze the data on {topic}.",
        expected_output="Summary of {topic} analysis.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])

    expected_placeholders = {"role_detail", "topic", "field"}
    actual_placeholders = crew.fetch_inputs()

    assert actual_placeholders == expected_placeholders, (
        f"Expected {expected_placeholders}, but got {actual_placeholders}"
    )


@pytest.mark.vcr()
def test_task_tools_preserve_code_execution_tools():
    """
    Test that task tools don't override code execution tools when allow_code_execution=True
    """
    class TestToolInput(BaseModel):
        """Input schema for TestTool."""

        query: str = Field(..., description="Query to process")

    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool that just returns the input"
        args_schema: type[BaseModel] = TestToolInput

        def _run(self, query: str) -> str:
            return f"Processed: {query}"

    # Mock Docker validation for the entire test
    with patch.object(Agent, "_validate_docker_installation"):
        # Create a programmer agent with code execution enabled
        programmer = Agent(
            role="Programmer",
            goal="Write code to solve problems.",
            backstory="You're a programmer who loves to solve problems with code.",
            allow_delegation=True,
            allow_code_execution=True,
        )

        # Create a code reviewer agent
        reviewer = Agent(
            role="Code Reviewer",
            goal="Review code for bugs and improvements",
            backstory="You're an experienced code reviewer who ensures code quality and best practices.",
            allow_delegation=True,
            allow_code_execution=True,
        )

    # Create a task with its own tools
    task = Task(
        description="Write a program to calculate fibonacci numbers.",
        expected_output="A working fibonacci calculator.",
        agent=programmer,
        tools=[TestTool()],
    )

    crew = Crew(
        agents=[programmer, reviewer],
        tasks=[task],
        process=Process.sequential,
    )

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Get the tools that were actually used in execution
        _, kwargs = mock_execute_sync.call_args
        used_tools = kwargs["tools"]

        # Verify all expected tools are present
        assert any(isinstance(tool, TestTool) for tool in used_tools), (
            "Task's TestTool should be present"
        )
        assert any(isinstance(tool, CodeInterpreterTool) for tool in used_tools), (
            "CodeInterpreterTool should be present"
        )
        assert any("delegate" in tool.name.lower() for tool in used_tools), (
            "Delegation tool should be present"
        )

        # Verify the total number of tools (TestTool + CodeInterpreter + 2 delegation tools)
        assert len(used_tools) == 4, (
            "Should have TestTool, CodeInterpreter, and 2 delegation tools"
        )


@pytest.mark.vcr()
def test_multimodal_flag_adds_multimodal_tools():
    """
    Test that an agent with multimodal=True automatically has multimodal tools added to the task execution.
    """
    # Create an agent that supports multimodal
    multimodal_agent = Agent(
        role="Multimodal Analyst",
        goal="Handle multiple media types (text, images, etc.).",
        backstory="You're an agent specialized in analyzing text, images, and other media.",
        allow_delegation=False,
        multimodal=True,  # crucial for adding the multimodal tool
    )

    # Create a dummy task
    task = Task(
        description="Describe what's in this image and generate relevant metadata.",
        expected_output="An image description plus any relevant metadata.",
        agent=multimodal_agent,
    )

    # Define a crew with the multimodal agent
    crew = Crew(agents=[multimodal_agent], tasks=[task], process=Process.sequential)

    mock_task_output = TaskOutput(
        description="Mock description", raw="mocked output", agent="mocked agent", messages=[]
    )

    # Mock execute_sync to verify the tools passed at runtime
    with patch.object(
        Task, "execute_sync", return_value=mock_task_output
    ) as mock_execute_sync:
        crew.kickoff()

        # Get the tools that were actually used in execution
        _, kwargs = mock_execute_sync.call_args
        used_tools = kwargs["tools"]

        # Check that the multimodal tool was added
        assert any(isinstance(tool, AddImageTool) for tool in used_tools), (
            "AddImageTool should be present when agent is multimodal"
        )

        # Verify we have exactly one tool (just the AddImageTool)
        assert len(used_tools) == 1, "Should only have the AddImageTool"


@pytest.mark.vcr()
def test_multimodal_agent_image_tool_handling():
    """
    Test that multimodal agents properly handle image tools in the CrewAgentExecutor
    """
    # Create a multimodal agent
    multimodal_agent = Agent(
        role="Image Analyst",
        goal="Analyze images and provide descriptions",
        backstory="You're an expert at analyzing and describing images.",
        allow_delegation=False,
        multimodal=True,
    )

    # Create a task that involves image analysis
    task = Task(
        description="Analyze this image and describe what you see.",
        expected_output="A detailed description of the image.",
        agent=multimodal_agent,
    )

    crew = Crew(agents=[multimodal_agent], tasks=[task])

    # Mock the image tool response
    mock_image_tool_result = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please analyze this image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/test-image.jpg",
                },
            },
        ],
    }

    # Create a mock task output for the final result
    mock_task_output = TaskOutput(
        description="Mock description",
        raw="A detailed analysis of the image",
        agent="Image Analyst",
        messages=[],
    )

    with patch.object(Task, "execute_sync") as mock_execute_sync:
        # Set up the mock to return our task output
        mock_execute_sync.return_value = mock_task_output

        # Execute the crew
        crew.kickoff()

        # Get the tools that were passed to execute_sync
        _, kwargs = mock_execute_sync.call_args
        tools = kwargs["tools"]

        # Verify the AddImageTool is present and properly configured
        image_tools = [tool for tool in tools if tool.name == "Add image to content"]
        assert len(image_tools) == 1, "Should have exactly one AddImageTool"

        # Test the tool's execution
        image_tool = image_tools[0]
        result = image_tool._run(
            image_url="https://example.com/test-image.jpg",
            action="Please analyze this image",
        )

        # Verify the tool returns the expected format
        assert result == mock_image_tool_result
        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image_url"


@pytest.mark.vcr()
def test_multimodal_agent_describing_image_successfully():
    """
    Test that a multimodal agent can process images without validation errors.
    This test reproduces the scenario from issue #2475.
    """
    llm = LLM(model="openai/gpt-4o", temperature=0.7)  # model with vision capabilities

    expert_analyst = Agent(
        role="Visual Quality Inspector",
        goal="Perform detailed quality analysis of product images",
        backstory="Senior quality control expert with expertise in visual inspection",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        multimodal=True,
    )

    inspection_task = Task(
        description="""
        Analyze the product image at https://www.us.maguireshoes.com/cdn/shop/files/FW24-Edito-Lucena-Distressed-01_1920x.jpg?v=1736371244 with focus on:
        1. Quality of materials
        2. Manufacturing defects
        3. Compliance with standards
        Provide a detailed report highlighting any issues found.
        """,
        expected_output="A detailed report highlighting any issues found",
        agent=expert_analyst,
    )

    crew = Crew(agents=[expert_analyst], tasks=[inspection_task])
    result = crew.kickoff()

    task_output = result.tasks_output[0]
    assert isinstance(task_output, TaskOutput)
    assert task_output.raw == result.raw


@pytest.mark.vcr()
def test_multimodal_agent_live_image_analysis():
    """
    Test that multimodal agents can analyze images through a real API call
    """
    # Create a multimodal agent
    image_analyst = Agent(
        role="Image Analyst",
        goal="Analyze images with high attention to detail",
        backstory="You're an expert at visual analysis, trained to notice and describe details in images.",
        allow_delegation=False,
        multimodal=True,
        verbose=True,
        llm="gpt-4o",
    )

    # Create a task for image analysis
    analyze_image = Task(
        description="""
        Analyze the provided image and describe what you see in detail.
        Focus on main elements, colors, composition, and any notable details.
        Image: {image_url}
        """,
        expected_output="A comprehensive description of the image contents.",
        agent=image_analyst,
    )

    # Create and run the crew
    crew = Crew(agents=[image_analyst], tasks=[analyze_image])

    # Execute with an image URL
    result = crew.kickoff(
        inputs={
            "image_url": "https://media.istockphoto.com/id/946087016/photo/aerial-view-of-lower-manhattan-new-york.jpg?s=612x612&w=0&k=20&c=viLiMRznQ8v5LzKTt_LvtfPFUVl1oiyiemVdSlm29_k="
        }
    )

    # Verify we got a meaningful response
    assert isinstance(result.raw, str)
    assert len(result.raw) > 100  # Expecting a detailed analysis
    assert "error" not in result.raw.lower()  # No error messages in response


@pytest.mark.vcr()
def test_crew_with_failing_task_guardrails():
    """Test that crew properly handles failing guardrails and retries with validation feedback."""

    def strict_format_guardrail(result: TaskOutput):
        """Validates that the output follows a strict format:
        - Must start with 'REPORT:'
        - Must end with 'END REPORT'
        """
        content = result.raw.strip()

        if not ("REPORT:" in content or "**REPORT:**" in content):
            return (
                False,
                "Output must start with 'REPORT:' no formatting, just the word REPORT",
            )

        if not ("END REPORT" in content or "**END REPORT**" in content):
            return (
                False,
                "Output must end with 'END REPORT' no formatting, just the word END REPORT",
            )

        return (True, content)

    researcher = Agent(
        role="Report Writer",
        goal="Create properly formatted reports",
        backstory="You're an expert at writing structured reports.",
    )

    task = Task(
        description="""Write a report about AI with exactly 3 key points.""",
        expected_output="A properly formatted report",
        agent=researcher,
        guardrail=strict_format_guardrail,
        guardrail_max_retries=3,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
    )

    result = crew.kickoff()

    # Verify the final output meets all format requirements
    content = result.raw.strip()
    assert content.startswith("REPORT:"), "Output should start with 'REPORT:'"
    assert content.endswith("END REPORT"), "Output should end with 'END REPORT'"

    # Verify task output
    task_output = result.tasks_output[0]
    assert isinstance(task_output, TaskOutput)
    assert task_output.raw == result.raw


@pytest.mark.vcr()
def test_crew_guardrail_feedback_in_context():
    """Test that guardrail feedback is properly appended to task context for retries."""

    def format_guardrail(result: TaskOutput):
        """Validates that the output contains a specific keyword."""
        if "IMPORTANT" not in result.raw:
            return (False, "Output must contain the keyword 'IMPORTANT'")
        return (True, result.raw)

    # Create execution contexts list to track contexts
    execution_contexts = []

    researcher = Agent(
        role="Writer",
        goal="Write content with specific keywords",
        backstory="You're an expert at following specific writing requirements.",
        allow_delegation=False,
    )

    task = Task(
        description="Write a short response.",
        expected_output="A response containing the keyword 'IMPORTANT'",
        agent=researcher,
        guardrail=format_guardrail,
        guardrail_max_retries=2,
    )

    crew = Crew(agents=[researcher], tasks=[task])

    with patch.object(Agent, "execute_task") as mock_execute_task:
        # Define side_effect to capture context and return different responses
        def side_effect(task, context=None, tools=None):
            execution_contexts.append(context if context else "")
            if len(execution_contexts) == 1:
                return "This is a test response"
            return "This is an IMPORTANT test response"

        mock_execute_task.side_effect = side_effect

        result = crew.kickoff()

    # Verify that we had multiple executions
    assert len(execution_contexts) > 1, "Task should have been executed multiple times"

    # Verify that the second execution included the guardrail feedback
    assert "Output must contain the keyword 'IMPORTANT'" in execution_contexts[1], (
        "Guardrail feedback should be included in retry context"
    )

    # Verify final output meets guardrail requirements
    assert "IMPORTANT" in result.raw, "Final output should contain required keyword"

    # Verify task retry count
    assert task.retry_count == 1, "Task should have been retried once"


@pytest.mark.vcr()
def test_before_kickoff_callback():
    @CrewBase
    class TestCrewClass:
        agents: list[BaseAgent]
        tasks: list[Task]

        agents_config = None
        tasks_config = None

        def __init__(self):
            self.inputs_modified = False

        @before_kickoff
        def modify_inputs(self, inputs):
            self.inputs_modified = True
            inputs["modified"] = True
            return inputs

        @agent
        def my_agent(self):
            return Agent(
                role="Test Agent",
                goal="Test agent goal",
                backstory="Test agent backstory",
            )

        @task
        def my_task(self):
            return Task(
                description="Test task description",
                expected_output="Test expected output",
                agent=self.my_agent(),
            )

        @crew
        def crew(self):
            return Crew(agents=self.agents, tasks=self.tasks)

    test_crew_instance = TestCrewClass()

    test_crew = test_crew_instance.crew()

    # Verify that the before_kickoff_callbacks are set
    assert len(test_crew.before_kickoff_callbacks) == 1

    # Prepare inputs
    inputs = {"initial": True}

    # Call kickoff
    test_crew.kickoff(inputs=inputs)

    # Check that the before_kickoff function was called and modified inputs
    assert test_crew_instance.inputs_modified
    assert inputs.get("modified")


@pytest.mark.vcr()
def test_before_kickoff_without_inputs():
    @CrewBase
    class TestCrewClass:
        agents_config = None
        tasks_config = None

        def __init__(self):
            self.inputs_modified = False
            self.received_inputs = None

        @before_kickoff
        def modify_inputs(self, inputs):
            self.inputs_modified = True
            inputs["modified"] = True
            self.received_inputs = inputs
            return inputs

        @agent
        def my_agent(self):
            return Agent(
                role="Test Agent",
                goal="Test agent goal",
                backstory="Test agent backstory",
            )

        @task
        def my_task(self):
            return Task(
                description="Test task description",
                expected_output="Test expected output",
                agent=self.my_agent(),
            )

        @crew
        def crew(self):
            return Crew(agents=self.agents, tasks=self.tasks)

    # Instantiate the class
    test_crew_instance = TestCrewClass()
    # Build the crew
    test_crew = test_crew_instance.crew()
    # Verify that the before_kickoff_callback is registered
    assert len(test_crew.before_kickoff_callbacks) == 1

    # Call kickoff without passing inputs
    test_crew.kickoff()

    # Check that the before_kickoff function was called
    assert test_crew_instance.inputs_modified

    # Verify that the inputs were initialized and modified inside the before_kickoff method
    assert test_crew_instance.received_inputs is not None
    assert test_crew_instance.received_inputs.get("modified") is True


@pytest.mark.vcr()
def test_crew_with_knowledge_sources_works_with_copy(researcher, writer):
    content = "Brandon's favorite color is red and he likes Mexican food."
    string_source = StringKnowledgeSource(content=content)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[Task(description="test", expected_output="test", agent=researcher)],
        knowledge_sources=[string_source],
    )
    crew_copy = crew.copy()

    assert crew_copy.knowledge_sources == crew.knowledge_sources
    assert len(crew_copy.agents) == len(crew.agents)
    assert len(crew_copy.tasks) == len(crew.tasks)

    assert len(crew_copy.tasks) == len(crew.tasks)


def test_crew_kickoff_for_each_works_with_manager_agent_copy():
    researcher = Agent(
        role="Researcher",
        goal="Conduct thorough research and analysis on AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently researching for a new client.",
        allow_delegation=False,
    )

    writer = Agent(
        role="Senior Writer",
        goal="Create compelling content about AI and AI agents",
        backstory="You're a senior writer, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently writing content for a new client.",
        allow_delegation=False,
    )

    # Define task
    task = Task(
        description="Generate a list of 5 interesting ideas for an article, then write one captivating paragraph for each idea that showcases the potential of a full article on this topic. Return the list of ideas with their paragraphs and your notes.",
        expected_output="5 bullet points, each with a paragraph and accompanying notes.",
    )

    # Define manager agent
    manager = Agent(
        role="Project Manager",
        goal="Efficiently manage the crew and ensure high-quality task completion",
        backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
        allow_delegation=True,
    )

    # Instantiate crew with a custom manager
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
        manager_agent=manager,
        process=Process.hierarchical,
        verbose=True,
    )

    crew_copy = crew.copy()
    assert crew_copy.manager_agent is not None
    assert crew_copy.manager_agent.id != crew.manager_agent.id
    assert crew_copy.manager_agent.role == crew.manager_agent.role
    assert crew_copy.manager_agent.goal == crew.manager_agent.goal


def test_crew_copy_with_memory():
    """Test that copying a crew with memory enabled does not raise validation errors and copies memory correctly."""
    agent = Agent(role="Test Agent", goal="Test Goal", backstory="Test Backstory")
    task = Task(description="Test Task", expected_output="Test Output", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], memory=True)

    original_short_term_id = (
        id(crew._short_term_memory) if crew._short_term_memory else None
    )
    original_long_term_id = (
        id(crew._long_term_memory) if crew._long_term_memory else None
    )
    original_entity_id = id(crew._entity_memory) if crew._entity_memory else None
    original_external_id = id(crew._external_memory) if crew._external_memory else None

    try:
        crew_copy = crew.copy()

        assert hasattr(crew_copy, "_short_term_memory"), (
            "Copied crew should have _short_term_memory"
        )
        assert crew_copy._short_term_memory is not None, (
            "Copied _short_term_memory should not be None"
        )
        assert id(crew_copy._short_term_memory) != original_short_term_id, (
            "Copied _short_term_memory should be a new object"
        )

        assert hasattr(crew_copy, "_long_term_memory"), (
            "Copied crew should have _long_term_memory"
        )
        assert crew_copy._long_term_memory is not None, (
            "Copied _long_term_memory should not be None"
        )
        assert id(crew_copy._long_term_memory) != original_long_term_id, (
            "Copied _long_term_memory should be a new object"
        )

        assert hasattr(crew_copy, "_entity_memory"), (
            "Copied crew should have _entity_memory"
        )
        assert crew_copy._entity_memory is not None, (
            "Copied _entity_memory should not be None"
        )
        assert id(crew_copy._entity_memory) != original_entity_id, (
            "Copied _entity_memory should be a new object"
        )

        if original_external_id:
            assert hasattr(crew_copy, "_external_memory"), (
                "Copied crew should have _external_memory"
            )
            assert crew_copy._external_memory is not None, (
                "Copied _external_memory should not be None"
            )
            assert id(crew_copy._external_memory) != original_external_id, (
                "Copied _external_memory should be a new object"
            )
        else:
            assert (
                not hasattr(crew_copy, "_external_memory")
                or crew_copy._external_memory is None
            ), "Copied _external_memory should be None if not originally present"

    except pydantic_core.ValidationError as e:
        if "Input should be an instance of" in str(e) and ("Memory" in str(e)):
            pytest.fail(
                f"Copying with memory raised Pydantic ValidationError, likely due to incorrect memory copy: {e}"
            )
        else:
            raise e  # Re-raise other validation errors
    except Exception as e:
        pytest.fail(f"Copying crew raised an unexpected exception: {e}")


def test_sets_parent_flow_when_outside_flow(researcher, writer):
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
    )
    assert crew.parent_flow is None


@pytest.mark.vcr()
def test_sets_parent_flow_when_inside_flow(researcher, writer):
    class MyFlow(Flow):
        @start()
        def start(self):
            return Crew(
                agents=[researcher, writer],
                process=Process.sequential,
                tasks=[
                    Task(
                        description="Task 1", expected_output="output", agent=researcher
                    ),
                    Task(description="Task 2", expected_output="output", agent=writer),
                ],
            )

    flow = MyFlow()
    result = flow.kickoff()
    assert result.parent_flow is flow


def test_reset_knowledge_with_no_crew_knowledge(researcher, writer):
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
    )

    with pytest.raises(RuntimeError) as excinfo:
        crew.reset_memories(command_type="knowledge")

        # Optionally, you can also check the error message
    assert "Crew Knowledge and Agent Knowledge memory system is not initialized" in str(
        excinfo.value
    )  # Replace with the expected message


def test_reset_knowledge_with_only_crew_knowledge(researcher, writer):
    mock_ks = MagicMock(spec=Knowledge)

    with patch.object(Crew, "reset_knowledge") as mock_reset_agent_knowledge:
        crew = Crew(
            agents=[researcher, writer],
            process=Process.sequential,
            tasks=[
                Task(description="Task 1", expected_output="output", agent=researcher),
                Task(description="Task 2", expected_output="output", agent=writer),
            ],
            knowledge=mock_ks,
        )

        crew.reset_memories(command_type="knowledge")
        mock_reset_agent_knowledge.assert_called_once_with([mock_ks])


def test_reset_knowledge_with_crew_and_agent_knowledge(researcher, writer):
    mock_ks_crew = MagicMock(spec=Knowledge)
    mock_ks_research = MagicMock(spec=Knowledge)
    mock_ks_writer = MagicMock(spec=Knowledge)

    researcher.knowledge = mock_ks_research
    writer.knowledge = mock_ks_writer

    with patch.object(Crew, "reset_knowledge") as mock_reset_agent_knowledge:
        crew = Crew(
            agents=[researcher, writer],
            process=Process.sequential,
            tasks=[
                Task(description="Task 1", expected_output="output", agent=researcher),
                Task(description="Task 2", expected_output="output", agent=writer),
            ],
            knowledge=mock_ks_crew,
        )

        crew.reset_memories(command_type="knowledge")
        mock_reset_agent_knowledge.assert_called_once_with(
            [mock_ks_crew, mock_ks_research, mock_ks_writer]
        )


def test_reset_knowledge_with_only_agent_knowledge(researcher, writer):
    mock_ks_research = MagicMock(spec=Knowledge)
    mock_ks_writer = MagicMock(spec=Knowledge)

    researcher.knowledge = mock_ks_research
    writer.knowledge = mock_ks_writer

    with patch.object(Crew, "reset_knowledge") as mock_reset_agent_knowledge:
        crew = Crew(
            agents=[researcher, writer],
            process=Process.sequential,
            tasks=[
                Task(description="Task 1", expected_output="output", agent=researcher),
                Task(description="Task 2", expected_output="output", agent=writer),
            ],
        )

        crew.reset_memories(command_type="knowledge")
        mock_reset_agent_knowledge.assert_called_once_with(
            [mock_ks_research, mock_ks_writer]
        )


def test_reset_agent_knowledge_with_no_agent_knowledge(researcher, writer):
    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
    )

    with pytest.raises(RuntimeError) as excinfo:
        crew.reset_memories(command_type="agent_knowledge")

    # Optionally, you can also check the error message
    assert "Agent Knowledge memory system is not initialized" in str(
        excinfo.value
    )  # Replace with the expected message


def test_reset_agent_knowledge_with_only_crew_knowledge(researcher, writer):
    mock_ks = MagicMock(spec=Knowledge)

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
        knowledge=mock_ks,
    )

    with pytest.raises(RuntimeError) as excinfo:
        crew.reset_memories(command_type="agent_knowledge")

    # Optionally, you can also check the error message
    assert "Agent Knowledge memory system is not initialized" in str(
        excinfo.value
    )  # Replace with the expected message


def test_reset_agent_knowledge_with_crew_and_agent_knowledge(researcher, writer):
    mock_ks_crew = MagicMock(spec=Knowledge)
    mock_ks_research = MagicMock(spec=Knowledge)
    mock_ks_writer = MagicMock(spec=Knowledge)

    researcher.knowledge = mock_ks_research
    writer.knowledge = mock_ks_writer

    with patch.object(Crew, "reset_knowledge") as mock_reset_agent_knowledge:
        crew = Crew(
            agents=[researcher, writer],
            process=Process.sequential,
            tasks=[
                Task(description="Task 1", expected_output="output", agent=researcher),
                Task(description="Task 2", expected_output="output", agent=writer),
            ],
            knowledge=mock_ks_crew,
        )

        crew.reset_memories(command_type="agent_knowledge")
        mock_reset_agent_knowledge.assert_called_once_with(
            [mock_ks_research, mock_ks_writer]
        )


def test_reset_agent_knowledge_with_only_agent_knowledge(researcher, writer):
    mock_ks_research = MagicMock(spec=Knowledge)
    mock_ks_writer = MagicMock(spec=Knowledge)

    researcher.knowledge = mock_ks_research
    writer.knowledge = mock_ks_writer

    with patch.object(Crew, "reset_knowledge") as mock_reset_agent_knowledge:
        crew = Crew(
            agents=[researcher, writer],
            process=Process.sequential,
            tasks=[
                Task(description="Task 1", expected_output="output", agent=researcher),
                Task(description="Task 2", expected_output="output", agent=writer),
            ],
        )

        crew.reset_memories(command_type="agent_knowledge")
        mock_reset_agent_knowledge.assert_called_once_with(
            [mock_ks_research, mock_ks_writer]
        )


def test_default_crew_name(researcher, writer):
    crew = Crew(
        agents=[researcher, writer],
        tasks=[
            Task(description="Task 1", expected_output="output", agent=researcher),
            Task(description="Task 2", expected_output="output", agent=writer),
        ],
    )
    assert crew.name == "crew"


@pytest.mark.vcr()
def test_ensure_exchanged_messages_are_propagated_to_external_memory():
    external_memory = ExternalMemory(storage=MagicMock())

    math_researcher = Agent(
        role="Researcher",
        goal="You research about math.",
        backstory="You're an expert in research and you love to learn new things.",
        allow_delegation=False,
    )

    task1 = Task(
        description="Research a topic to teach a kid aged 6 about math.",
        expected_output="A topic, explanation, angle, and examples.",
        agent=math_researcher,
    )

    crew = Crew(
        agents=[math_researcher],
        tasks=[task1],
        external_memory=external_memory,
    )

    with patch.object(
        ExternalMemory, "save", return_value=None
    ) as external_memory_save:
        crew.kickoff()

    external_memory_save.assert_called_once()

    call_args = external_memory_save.call_args

    assert "value" in call_args.kwargs or len(call_args.args) > 0
    assert "metadata" in call_args.kwargs or len(call_args.args) > 1

    if "metadata" in call_args.kwargs:
        metadata = call_args.kwargs["metadata"]
    else:
        metadata = call_args.args[1]

    assert "description" in metadata
    assert "messages" in metadata
    assert isinstance(metadata["messages"], list)
    assert len(metadata["messages"]) >= 2

    messages = metadata["messages"]
    assert messages[0]["role"] == "system"
    assert "Researcher" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Research a topic to teach a kid aged 6 about math" in messages[1]["content"]
