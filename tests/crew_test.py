"""Test Agent creation and execution basic functionality."""

import json

import pydantic_core
import pytest

from crewai.agent import Agent
from crewai.agents.cache import CacheHandler
from crewai.crew import Crew
from crewai.process import Process
from crewai.task import Task
from crewai.utilities import Logger, RPMController

ceo = Agent(
    role="CEO",
    goal="Make sure the writers in your company produce amazing content.",
    backstory="You're an long time CEO of a content creation agency with a Senior Writer on the team. You're now working on a new project and want to make sure the content produced is amazing.",
    allow_delegation=True,
)

researcher = Agent(
    role="Researcher",
    goal="Make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
    allow_delegation=False,
)

writer = Agent(
    role="Senior Writer",
    goal="Write the best content about AI and AI agents.",
    backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
    allow_delegation=False,
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
                    "agent": "Senior Researcher",
                },
                {
                    "description": "Write a 1 amazing paragraph highlight for each idead that showcases how good an article about this topic could be, check references if necessary or search for more content but make sure it's unique, interesting and well written. Return the list of ideas with their paragraph and your notes.",
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_creation():
    tasks = [
        Task(
            description="Give me a list of 5 interesting ideas to explore for na article, what makes them unique and interesting.",
            agent=researcher,
        ),
        Task(
            description="Write a 1 amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
            agent=writer,
        ),
    ]

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=tasks,
    )

    assert (
        crew.kickoff()
        == '1. "The Role of AI in Predictive Analysis"\nHighlight: AI is revolutionizing the way we understand and utilize data through predictive analysis. Complex algorithms can sift through vast amounts of information, predict future trends and assist businesses in making informed decisions. The article will delve into the intricate workings of AI in predictive analysis and how it is shaping industries from healthcare to finance.\n\nNotes: This topic will focus on the business aspect of AI and its transformative role in data analysis. Case studies from different industries can be used to illustrate the impact of AI in predictive analysis.\n\n2. "The Intersection of AI and Quantum Computing"\nHighlight: As we stand at the crossroads of AI and quantum computing, there’s an unprecedented potential for breakthroughs in processing speed and problem-solving capabilities. This article will explore this exciting intersection, revealing how the fusion of these two technologies can push the boundaries of what\'s possible.\n\nNotes: The article will provide a detailed overview of quantum computing and how its integration with AI can revolutionize various sectors. Real-world applications and future predictions will be included.\n\n3. "AI for Sustainable Development"\nHighlight: In an era where sustainability is a global priority, AI is emerging as a powerful tool in progressing towards this goal. From optimizing resource use to monitoring environmental changes, AI\'s role in sustainable development is multifaceted and transformative. This article will shed light on how AI is being utilized to promote a more sustainable future.\n\nNotes: This topic will delve into the environmental aspect of AI and its potential in promoting sustainable development. Examples of AI applications in different environmental contexts will be provided.\n\n4. "Ethical Implications of AI"\nHighlight: As AI permeates our society, it brings along a host of ethical dilemmas. From privacy concerns to accountability, the ethical implications of AI are as complex as they are critical. This article will take a deep dive into the ethical landscape of AI, exploring the pressing issues and potential solutions.\n\nNotes: This topic will take a philosophical and ethical approach, discussing the moral implications of AI use and how they can be mitigated. It will include a wide range of perspectives from experts in the field.\n\n5. "AI in Art and Creativity"\nHighlight: The world of art is no stranger to the transformative power of AI. From creating original artworks to enhancing creative processes, AI is redefining the boundaries of art and creativity. This article will take you on a journey through the fascinating intersection of AI and creativity, showcasing the revolutionary impact of this technology in the art world.\n\nNotes: This article will explore the artistic side of AI, discussing how it\'s being used in various creative fields. It will feature interviews with artists and creators who are harnessing the power of AI in their work.'
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_hierarchical_process():
    from langchain_openai import ChatOpenAI

    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tasks=[task],
    )

    assert (
        crew.kickoff()
        == "Here are the five interesting ideas for our next article along with a captivating paragraph for each:\n\n1. 'AI and Climate Change: A New Hope for Sustainability':\nIn a world where climate change is a pressing concern, Artificial Intelligence (AI) offers a glimmer of hope. This article will delve into how AI's predictive capabilities and data analysis can aid in sustainability efforts, from optimizing energy consumption to predicting extreme weather patterns. Through real-world examples and expert insights, we'll explore the innovative solutions AI is bringing to the fight against climate change.\n\n2. 'AI in Art: How Neural Networks are Revolutionizing the Artistic Landscape':\nArtificial Intelligence is not just for the tech-savvy; it's making waves in the art world too. This article will unveil how AI and Neural Networks are transforming the artistic landscape, creating a new genre of AI-art. From AI that can replicate the style of famous artists to AI that creates entirely original pieces, we will delve into this fascinating intersection of technology and creativity.\n\n3. 'The Role of AI in the Post-Covid World':\nThe global pandemic has drastically altered our world, and AI has played a pivotal role in this transformation. In this article, we'll explore how AI has been instrumental in everything from predicting the virus's spread to accelerating vaccine development. We'll also look ahead to the post-Covid world, investigating the lasting changes that AI will bring about in our societies.\n\n4. 'Demystifying AI: Breaking Down Complex AI Concepts for the Everyday Reader':\nArtificial Intelligence can seem like a complex and intimidating subject, but it doesn't have to be. This article aims to demystify AI, breaking down complex concepts into understandable nuggets of information. Whether you're an AI novice or a tech enthusiast, this article will enrich your understanding of AI and its impact on our lives.\n\n5. 'The Ethical Dilemmas of AI: Balancing Innovation and Humanity':\nAs AI continues to advance, it brings along a host of ethical dilemmas. This article will delve into the heart of these issues, discussing the balance between innovation and humanity. From the potential for bias in AI algorithms to the implications of autonomous machines, we'll explore the ethical implications of AI in our society."
    )


def test_manager_llm_requirement_for_hierarchical_process():
    task = Task(
        description="Come up with a list of 5 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
    )

    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        Crew(
            agents=[researcher, writer],
            process=Process.hierarchical,
            tasks=[task],
        )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_with_delegating_agents():
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

    assert (
        crew.kickoff()
        == "In today's technological landscape, Artificial Intelligence (AI) agents have emerged as key players in shaping the future of various industries. These agents, which are essentially computer programs that can learn, adapt, and operate autonomously, are a testament to the rapidly evolving capabilities of AI. They are the harbingers of a new era, where machines can mimic human intelligence, and in some cases, even surpass it.\n\nAI agents are transforming the way we engage with technology, enabling a more personalized and efficient user experience. They are extensively used in areas like customer service, where chatbots can handle customer inquiries without human intervention. They have revolutionized sectors like healthcare, where AI agents can analyze patient data to predict health trends and provide personalized treatment recommendations. \n\nHowever, as AI agents continue to evolve, they also pose significant ethical and regulatory challenges. There are concerns about privacy, bias, and the potential misuse of these technologies. As a society, it's crucial to establish norms and regulations that ensure the responsible use of AI agents, balancing their benefits with potential risks.\n\nIn conclusion, AI agents are a transformative technology that is reshaping our world. The challenges they present are complex, but the opportunities they offer are immense. As we continue to explore and understand this technology, we can harness its potential to create a more efficient, personalized, and intelligent future."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_verbose_output(capsys):
    tasks = [
        Task(description="Research AI advancements.", agent=researcher),
        Task(description="Write about AI in healthcare.", agent=writer),
    ]

    crew = Crew(
        agents=[researcher, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = [
        "[DEBUG]: Working Agent: Researcher",
        "[INFO]: Starting Task: Research AI advancements.",
        "[DEBUG]: [Researcher] Task output:",
        "[DEBUG]: Working Agent: Senior Writer",
        "[INFO]: Starting Task: Write about AI in healthcare.",
        "[DEBUG]: [Senior Writer] Task output:",
    ]

    for expected_string in expected_strings:
        assert expected_string in captured.out

    # Now test with verbose set to False
    crew._logger = Logger(verbose_level=False)
    crew.kickoff()
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_verbose_levels_output(capsys):
    tasks = [Task(description="Write about AI advancements.", agent=researcher)]

    crew = Crew(agents=[researcher], tasks=tasks, process=Process.sequential, verbose=1)

    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = ["Working Agent: Researcher", "[Researcher] Task output:"]

    for expected_string in expected_strings:
        assert expected_string in captured.out

    # Now test with verbose set to 2
    crew._logger = Logger(verbose_level=2)
    crew.kickoff()
    captured = capsys.readouterr()
    expected_strings = [
        "Working Agent: Researcher",
        "Starting Task: Write about AI advancements.",
        "[Researcher] Task output:",
    ]

    for expected_string in expected_strings:
        assert expected_string in captured.out


@pytest.mark.vcr(filter_headers=["authorization"])
def test_cache_hitting_between_agents():
    from unittest.mock import call, patch

    from langchain.tools import tool

    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    tasks = [
        Task(
            description="What is 2 tims 6? Return only the number.",
            tools=[multiplier],
            agent=ceo,
        ),
        Task(
            description="What is 2 times 6? Return only the number.",
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
        # Check if read was called with the expected arguments
        expected_calls = [
            call(tool="multiplier", input={"first_number": 2, "second_number": 6}),
            call(tool="multiplier", input={"first_number": 2, "second_number": 6}),
        ]
        read.assert_has_calls(expected_calls, any_order=False)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_api_calls_throttling(capsys):
    from unittest.mock import patch

    from langchain.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def get_final_answer(anything) -> float:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return 42

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        max_iter=5,
        allow_delegation=False,
        verbose=True,
        llm=ChatOpenAI(model="gpt-4-0125-preview"),
    )

    task = Task(
        description="Don't give a Final Answer, instead keep using the `get_final_answer` tool.",
        tools=[get_final_answer],
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], max_rpm=2, verbose=2)

    with patch.object(RPMController, "_wait_for_next_minute") as moveon:
        moveon.return_value = True
        crew.kickoff()
        captured = capsys.readouterr()
        assert "Max RPM reached, waiting for next minute to start." in captured.out
        moveon.assert_called()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_full_ouput():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
        verbose=True,
    )

    task1 = Task(
        description="just say hi!",
        agent=agent,
    )
    task2 = Task(
        description="just say hello!",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task1, task2], full_output=True)

    result = crew.kickoff()
    assert result == {
        "final_output": "Hello!",
        "tasks_outputs": [task1.output, task2.output],
    }


def test_agents_rpm_is_never_set_if_crew_max_RPM_is_not_set():
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description="just say hi!",
        agent=agent,
    )

    Crew(agents=[agent], tasks=[task], verbose=2)

    assert agent._rpm_controller is None


def test_async_task_execution():
    import threading
    from unittest.mock import patch

    from crewai.tasks.task_output import TaskOutput

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
        context=[list_ideas, list_important_history],
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.sequential,
        tasks=[list_ideas, list_important_history, write_article],
    )

    with patch.object(Agent, "execute_task") as execute:
        execute.return_value = "ok"
        with patch.object(threading.Thread, "start") as start:
            thread = threading.Thread(target=lambda: None, args=()).start()
            start.return_value = thread
            with patch.object(threading.Thread, "join", wraps=thread.join()) as join:
                list_ideas.output = TaskOutput(
                    description="A 4 paragraph article about AI.", raw_output="ok"
                )
                list_important_history.output = TaskOutput(
                    description="A 4 paragraph article about AI.", raw_output="ok"
                )
                crew.kickoff()
                start.assert_called()
                join.assert_called()


def test_set_agents_step_callback():
    from unittest.mock import patch

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
    from unittest.mock import patch

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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_function_calling_llm():
    from unittest.mock import patch

    from langchain.tools import tool
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    with patch.object(llm.client, "create", wraps=llm.client.create) as private_mock:

        @tool
        def learn_about_AI(topic) -> float:
            """Useful for when you need to learn about AI to write an paragraph about it."""
            return "AI is a very broad field."

        agent1 = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            llm=ChatOpenAI(model="gpt-4-0125-preview"),
            tools=[learn_about_AI],
        )

        essay = Task(
            description="Write and then review an small paragraph on AI until it's AMAZING",
            agent=agent1,
        )
        tasks = [essay]
        crew = Crew(agents=[agent1], tasks=tasks, function_calling_llm=llm)
        crew.kickoff()
        private_mock.assert_called()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_task_with_no_arguments():
    from langchain.tools import tool

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
        description="Look at the available data nd give me a sense on the total number of sales.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task])

    result = crew.kickoff()
    assert result == "The total number of sales from January to May is 75."


def test_delegation_is_not_enabled_if_there_are_only_one_agent():
    from unittest.mock import patch

    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=True,
    )

    task = Task(
        description="Look at the available data nd give me a sense on the total number of sales.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task])

    with patch.object(Task, "execute") as execute:
        execute.return_value = "ok"
        crew.kickoff()
        assert task.tools == []


@pytest.mark.vcr(filter_headers=["authorization"])
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
    assert result == "Howdy!"
    assert len(agent.tools) == 0


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_usage_metrics_are_captured_for_sequential_process():
    agent = Agent(
        role="Researcher",
        goal="Be super empathetic.",
        backstory="You're love to sey howdy.",
        allow_delegation=False,
    )

    task = Task(description="say howdy", expected_output="Howdy!", agent=agent)

    crew = Crew(agents=[agent], tasks=[task])

    result = crew.kickoff()
    assert result == "Howdy!"
    assert crew.usage_metrics == {
        "completion_tokens": 8,
        "prompt_tokens": 103,
        "successful_requests": 1,
        "total_tokens": 111,
    }


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_usage_metrics_are_captured_for_hierarchical_process():
    from langchain_openai import ChatOpenAI

    agent = Agent(
        role="Researcher",
        goal="Be super empathetic.",
        backstory="You're love to sey howdy.",
        allow_delegation=False,
    )

    task = Task(description="say howdy", expected_output="Howdy!")

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(temperature=0, model="gpt-4"),
    )

    result = crew.kickoff()
    assert result == "Howdy!"
    assert crew.usage_metrics == {
        "total_tokens": 1365,
        "prompt_tokens": 1256,
        "completion_tokens": 109,
        "successful_requests": 3,
    }


def test_crew_inputs_interpolate_both_agents_and_tasks():
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="{points} bullet points about {topic}.",
    )

    crew = Crew(agents=[agent], tasks=[task], inputs={"topic": "AI", "points": 5})

    assert crew.tasks[0].description == "Give me an analysis around AI."
    assert crew.tasks[0].expected_output == "5 bullet points about AI."
    assert crew.agents[0].role == "AI Researcher"
    assert crew.agents[0].goal == "Express hot takes on AI."
    assert crew.agents[0].backstory == "You have a lot of experience with AI."


def test_crew_inputs_interpolate_both_agents_and_tasks():
    from unittest.mock import patch

    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )

    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="{points} bullet points about {topic}.",
    )

    with patch.object(Agent, "interpolate_inputs") as interpolate_agent_inputs:
        with patch.object(Task, "interpolate_inputs") as interpolate_task_inputs:
            interpolate_agent_inputs.return_value = None
            interpolate_task_inputs.return_value = None
            Crew(agents=[agent], tasks=[task], inputs={"topic": "AI", "points": 5})
            interpolate_agent_inputs.assert_called()
            interpolate_task_inputs.assert_called()
