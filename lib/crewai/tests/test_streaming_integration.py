"""Integration tests for streaming with real LLM interactions using cassettes."""

import pytest

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start
from crewai.types.streaming import CrewStreamingOutput, FlowStreamingOutput


@pytest.fixture
def researcher() -> Agent:
    """Create a researcher agent for testing."""
    return Agent(
        role="Research Analyst",
        goal="Gather comprehensive information on topics",
        backstory="You are an experienced researcher with excellent analytical skills.",
        allow_delegation=False,
    )


@pytest.fixture
def simple_task(researcher: Agent) -> Task:
    """Create a simple research task."""
    return Task(
        description="Research the latest developments in {topic}",
        expected_output="A brief summary of recent developments",
        agent=researcher,
    )


class TestStreamingCrewIntegration:
    """Integration tests for crew streaming that match documentation examples."""

    @pytest.mark.vcr()
    def test_basic_crew_streaming_from_docs(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test basic streaming example from documentation."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            stream=True,
            verbose=False,
        )

        streaming = crew.kickoff(inputs={"topic": "artificial intelligence"})

        assert isinstance(streaming, CrewStreamingOutput)

        chunks = []
        for chunk in streaming:
            chunks.append(chunk.content)

        assert len(chunks) > 0

        result = streaming.result
        assert result.raw is not None
        assert len(result.raw) > 0

    @pytest.mark.vcr()
    def test_streaming_with_chunk_context_from_docs(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test streaming with chunk context example from documentation."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            stream=True,
            verbose=False,
        )

        streaming = crew.kickoff(inputs={"topic": "AI"})

        chunk_contexts = []
        for chunk in streaming:
            chunk_contexts.append(
                {
                    "task_name": chunk.task_name,
                    "task_index": chunk.task_index,
                    "agent_role": chunk.agent_role,
                    "content": chunk.content,
                    "type": chunk.chunk_type,
                }
            )

        assert len(chunk_contexts) > 0
        assert all("agent_role" in ctx for ctx in chunk_contexts)

        result = streaming.result
        assert result is not None

    @pytest.mark.vcr()
    def test_streaming_properties_from_docs(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test streaming properties example from documentation."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            stream=True,
            verbose=False,
        )

        streaming = crew.kickoff(inputs={"topic": "AI"})

        for _ in streaming:
            pass

        assert streaming.is_completed is True
        full_text = streaming.get_full_text()
        assert len(full_text) > 0
        assert len(streaming.chunks) > 0

        result = streaming.result
        assert result.raw is not None

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_async_streaming_from_docs(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test async streaming example from documentation."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            stream=True,
            verbose=False,
        )

        streaming = await crew.kickoff_async(inputs={"topic": "AI"})

        assert isinstance(streaming, CrewStreamingOutput)

        chunks = []
        async for chunk in streaming:
            chunks.append(chunk.content)

        assert len(chunks) > 0

        result = streaming.result
        assert result.raw is not None

    @pytest.mark.vcr()
    def test_kickoff_for_each_streaming_from_docs(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test kickoff_for_each streaming example from documentation."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            stream=True,
            verbose=False,
        )

        inputs_list = [{"topic": "AI in healthcare"}, {"topic": "AI in finance"}]

        streaming_outputs = crew.kickoff_for_each(inputs=inputs_list)

        assert len(streaming_outputs) == 2
        assert all(isinstance(s, CrewStreamingOutput) for s in streaming_outputs)

        results = []
        for streaming in streaming_outputs:
            for _ in streaming:
                pass

            result = streaming.result
            results.append(result)

        assert len(results) == 2
        assert all(r.raw is not None for r in results)


class TestStreamingFlowIntegration:
    """Integration tests for flow streaming that match documentation examples."""

    @pytest.mark.vcr()
    def test_basic_flow_streaming_from_docs(self) -> None:
        """Test basic flow streaming example from documentation."""

        class ResearchFlow(Flow):
            stream = True

            @start()
            def research_topic(self) -> str:
                researcher = Agent(
                    role="Research Analyst",
                    goal="Research topics thoroughly",
                    backstory="Expert researcher with analytical skills",
                    allow_delegation=False,
                )

                task = Task(
                    description="Research AI trends and provide insights",
                    expected_output="Detailed research findings",
                    agent=researcher,
                )

                crew = Crew(
                    agents=[researcher],
                    tasks=[task],
                    stream=True,
                    verbose=False,
                )

                streaming = crew.kickoff()
                for _ in streaming:
                    pass
                return streaming.result.raw

        flow = ResearchFlow()

        streaming = flow.kickoff()

        assert isinstance(streaming, FlowStreamingOutput)

        chunks = []
        for chunk in streaming:
            chunks.append(chunk.content)

        assert len(chunks) > 0

        result = streaming.result
        assert result is not None

    @pytest.mark.vcr()
    def test_flow_streaming_properties_from_docs(self) -> None:
        """Test flow streaming properties example from documentation."""

        class SimpleFlow(Flow):
            stream = True

            @start()
            def execute(self) -> str:
                return "Flow result"

        flow = SimpleFlow()
        streaming = flow.kickoff()

        for _ in streaming:
            pass

        assert streaming.is_completed is True
        streaming.get_full_text()
        assert len(streaming.chunks) >= 0

        result = streaming.result
        assert result is not None

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_async_flow_streaming_from_docs(self) -> None:
        """Test async flow streaming example from documentation."""

        class AsyncResearchFlow(Flow):
            stream = True

            @start()
            def research(self) -> str:
                researcher = Agent(
                    role="Researcher",
                    goal="Research topics",
                    backstory="Expert researcher",
                    allow_delegation=False,
                )

                task = Task(
                    description="Research AI",
                    expected_output="Research findings",
                    agent=researcher,
                )

                crew = Crew(agents=[researcher], tasks=[task], stream=True, verbose=False)
                streaming = crew.kickoff()
                for _ in streaming:
                    pass
                return streaming.result.raw

        flow = AsyncResearchFlow()

        streaming = await flow.kickoff_async()

        assert isinstance(streaming, FlowStreamingOutput)

        chunks = []
        async for chunk in streaming:
            chunks.append(chunk.content)

        result = streaming.result
        assert result is not None
