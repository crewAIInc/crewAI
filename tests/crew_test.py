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
        == """Here are the five interesting ideas for articles with their respective highlights:\n\n1. The Role of AI in Climate Change: As the world grapples with the existential threat of climate change, artificial intelligence (AI) has emerged as a powerful ally in our battle against it. The article will explore how AI is being used to predict weather patterns, optimize renewable energy sources, and even capture and reduce greenhouse emissions. This novel intersection of technology and environment could hold the key to a sustainable future, making this a must-read for anyone interested in the potential of AI to transform our world.\n\n2. AI and Mental Health: With the increasing prevalence of mental health issues worldwide, innovative solutions are needed more than ever. This article will delve into the cutting-edge domain of AI and mental health, exploring how machine learning algorithms are helping to diagnose conditions, personalize treatments, and even predict the onset of mental disorders. This exploration of AI's potential in mental health not only sheds light on the future of healthcare but also opens a dialogue on the ethical considerations involved.\n\n3. The Ethical Implications of AI: As AI continues to permeate our lives, it brings with it a host of ethical considerations. This article will unravel the complex ethical terrain of AI, from issues of privacy and consent to its potential for bias and discrimination. By diving into the philosophical underpinnings of AI and its societal implications, this article will provoke thought and stimulate discussion on how we can ensure a fair and equitable AI-enabled future.\n\n4. How AI is Revolutionizing E-commerce: In the fiercely competitive world of e-commerce, AI is proving to be a game-changer. This article will take you on a journey through the world of AI-enhanced e-commerce, showcasing how machine learning algorithms are optimizing logistics, personalizing shopping experiences, and even predicting consumer behavior. This deep dive into AI's transformative impact on e-commerce is a must-read for anyone interested in the future of business and technology.\n\n5. AI in Space Exploration: The final frontier of space exploration is being redefined by the advent of AI. This article will take you on an interstellar journey through the role of AI in space exploration, from autonomous spacecraft navigation to the search for extraterrestrial life. By peering into the cosmos through the lens of AI, this article offers a glimpse into the future of space exploration and the infinite possibilities that AI holds."""
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
        == "The Senior Writer has produced a fantastic 4 paragraph article on AI:\n\n\"Artificial Intelligence, or AI, is often considered the stuff of science fiction, but it is very much a reality in today's world. In simplest terms, AI is a branch of computer science that aims to create machines that mimic human intelligence - think self-driving cars, voice assistants like Siri or Alexa, even your Netflix recommendations. These are all examples of AI in action, silently making our lives easier and more efficient.\n\nThe applications of AI are as vast as our imagination. In healthcare, AI is used to predict diseases and personalize patient care. In finance, algorithms can analyze market trends and make investment decisions. The education sector uses AI to customize learning and identify areas where students need help. Even in creative fields like music and art, AI is making its mark by creating new pieces that are hard to distinguish from those made by humans.\n\nAI's potential for the future is staggering. As technology advances, so too does the complexity and capabilities of AI. It's predicted that AI will play a significant role in tackling some of humanity's biggest challenges, such as climate change and global health crises. Imagine AI systems predicting natural disasters with enough time for us to take preventative measures, or developing new, effective treatments for diseases through data analysis.\n\nHowever, this brave new world does not come without its challenges. Ethical issues are at the forefront, with concerns over privacy and the potential misuse of AI. There's also the question of job displacement due to automation, and the need for laws and regulations to keep pace with this rapidly advancing technology. Despite these hurdles, the promise of AI and its ability to transform our world is an exciting prospect, one that we are only just beginning to explore.\""
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

    @tool
    def get_final_answer(numbers) -> float:
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
                    description="A 4 paragraph article about AI.", result="ok"
                )
                list_important_history.output = TaskOutput(
                    description="A 4 paragraph article about AI.", result="ok"
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

    llm = ChatOpenAI(model="gpt-3.5")

    with patch.object(llm.client, "create", wraps=llm.client.create) as private_mock:

        @tool
        def learn_about_AI(topic) -> float:
            """Useful for when you need to learn about AI to write an paragraph about it."""
            return "AI is a very broad field."

        agent1 = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            tools=[learn_about_AI],
        )

        essay = Task(
            description="Write and then review an small paragraph on AI until it's AMAZING",
            agent=agent1,
        )
        tasks = [essay]
        print(agent1.function_calling_llm)
        crew = Crew(agents=[agent1], tasks=tasks, function_calling_llm=llm)
        print(agent1.function_calling_llm)
        crew.kickoff()
        private_mock.assert_called()
