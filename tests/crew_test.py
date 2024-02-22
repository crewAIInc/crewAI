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
        == '1. "The Role of AI in Predicting and Managing Pandemics"\nHighlight: \nIn an era where global health crises can emerge from any corner of the world, the role of AI in predicting and managing pandemics has never been more critical. Through intelligent data gathering and predictive analytics, AI can potentially identify the onset of pandemics before they reach critical mass, offering a proactive solution to a reactive problem. This article explores the intersection of AI and epidemiology, delving into how this cutting-edge technology is revolutionizing our approach to global health crises.\n\n2. "AI and the Future of Work: Will Robots Take Our Jobs?"\nHighlight: \nThe rise of AI has sparked both excitement and apprehension about the future of work. Will robots replace us, or will they augment our capabilities? This article delves into the heart of this controversial issue, examining the potential of AI to disrupt job markets, transform industries, and redefine the concept of work. It\'s not just a question of job security—it\'s a discussion about the kind of world we want to live in.\n\n3. "AI in Art and Creativity: A New Frontier in Innovation"\nHighlight: \nArt and creativity, once seen as the exclusive domain of human expression, are being redefined by the advent of AI. From algorithmic compositions to AI-assisted design, this article explores the burgeoning field of AI in art and creativity. It\'s a journey into a new frontier of innovation, one where the lines between human creativity and artificial intelligence blur into an exciting, uncharted territory.\n\n4. "Ethics in AI: Balancing Innovation with Responsibility"\nHighlight: \nAs AI continues to permeate every facet of our lives, questions about its ethical implications grow louder. This article invites readers into a thoughtful exploration of the moral landscape of AI. It challenges us to balance the relentless pursuit of innovation with the weighty responsibilities that come with it, asking: How can we harness the power of AI without losing sight of our human values?\n\n5. "AI in Education: Personalizing Learning for the Next Generation"\nHighlight: \nEducation is poised for a transformation as AI enters the classroom, promising a future where learning is personalized, not generalized. This article delves into how AI can tailor educational experiences to individual learning styles, making education more effective and accessible. It\'s a glimpse into a future where AI is not just a tool for learning, but an active participant in shaping the educational journey of the next generation.'
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
        == """Here are the 5 unique and interesting ideas for articles along with a highlight paragraph for each:\n\n1) The Future of AI and Machine Learning: A deeper look into the future of AI and machine learning, revealing the potential of both and their implications on society. The article will provide an informed vision of the future, addressing the possibilities that AI and machine learning could bring to our daily lives, from healthcare to education, and the challenges we might face.\n\n2) Startups Revolutionizing Traditional Industries with Tech: This article will narrate the journey of game-changing startups that are transforming traditional industries with innovative technology. It will delve into their stories, exploring how they leverage technology to disrupt the status quo, the hurdles they've overcome, and the impact they're making.\n\n3) Personal Development in the Age of Technology: In this article, we will explore how technology has changed the landscape of personal development. We will cover how digital tools and platforms are empowering individuals to learn, grow, and achieve their goals faster than ever before.\n\n4) Ethical Issues in Software Engineering: This article will investigate the ethical dilemmas that are arising in the realm of software engineering. It will discuss the moral implications of new technologies, the responsibilities of software engineers, and the need for a robust code of ethics in this rapidly evolving field.\n\n5) Entrepreneurship in the Digital Era: In this piece, we will delve into the role of digital technology in shaping the entrepreneurial landscape. We will discuss how the digital era has given rise to new entrepreneurial opportunities, the challenges that come with it, and the skills required to thrive in this new era."""
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
