import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict
from unittest.mock import patch

import pytest
from crewai import Agent, Crew, Task
from crewai.utilities.crew.crew_context import get_crew_context


@pytest.fixture
def simple_agent_factory():
    def create_agent(name: str) -> Agent:
        return Agent(
            role=f"{name} Agent",
            goal=f"Complete {name} task",
            backstory=f"I am agent for {name}",
        )

    return create_agent


@pytest.fixture
def simple_task_factory():
    def create_task(name: str, callback: Callable = None) -> Task:
        return Task(
            description=f"Task for {name}", expected_output="Done", callback=callback
        )

    return create_task


@pytest.fixture
def crew_factory(simple_agent_factory, simple_task_factory):
    def create_crew(name: str, task_callback: Callable = None) -> Crew:
        agent = simple_agent_factory(name)
        task = simple_task_factory(name, callback=task_callback)
        task.agent = agent

        return Crew(agents=[agent], tasks=[task], verbose=False)

    return create_crew


class TestCrewThreadSafety:
    @patch("crewai.Agent.execute_task")
    def test_parallel_crews_thread_safety(self, mock_execute_task, crew_factory):
        mock_execute_task.return_value = "Task completed"
        num_crews = 5

        def run_crew_with_context_check(crew_id: str) -> Dict[str, Any]:
            results = {"crew_id": crew_id, "contexts": []}

            def check_context_task(output):
                context = get_crew_context()
                results["contexts"].append(
                    {
                        "stage": "task_callback",
                        "crew_id": context.id if context else None,
                        "crew_key": context.key if context else None,
                        "thread": threading.current_thread().name,
                    }
                )
                return output

            context_before = get_crew_context()
            results["contexts"].append(
                {
                    "stage": "before_kickoff",
                    "crew_id": context_before.id if context_before else None,
                    "thread": threading.current_thread().name,
                }
            )

            crew = crew_factory(crew_id, task_callback=check_context_task)
            output = crew.kickoff()

            context_after = get_crew_context()
            results["contexts"].append(
                {
                    "stage": "after_kickoff",
                    "crew_id": context_after.id if context_after else None,
                    "thread": threading.current_thread().name,
                }
            )

            results["crew_uuid"] = str(crew.id)
            results["output"] = output.raw

            return results

        with ThreadPoolExecutor(max_workers=num_crews) as executor:
            futures = []
            for i in range(num_crews):
                future = executor.submit(run_crew_with_context_check, f"crew_{i}")
                futures.append(future)

            results = [f.result() for f in futures]

        for result in results:
            crew_uuid = result["crew_uuid"]

            before_ctx = next(
                ctx for ctx in result["contexts"] if ctx["stage"] == "before_kickoff"
            )
            assert before_ctx["crew_id"] is None, (
                f"Context should be None before kickoff for {result['crew_id']}"
            )

            task_ctx = next(
                ctx for ctx in result["contexts"] if ctx["stage"] == "task_callback"
            )
            assert task_ctx["crew_id"] == crew_uuid, (
                f"Context mismatch during task for {result['crew_id']}"
            )

            after_ctx = next(
                ctx for ctx in result["contexts"] if ctx["stage"] == "after_kickoff"
            )
            assert after_ctx["crew_id"] is None, (
                f"Context should be None after kickoff for {result['crew_id']}"
            )

            thread_name = before_ctx["thread"]
            assert "ThreadPoolExecutor" in thread_name, (
                f"Should run in thread pool for {result['crew_id']}"
            )

    @pytest.mark.asyncio
    @patch("crewai.Agent.execute_task")
    async def test_async_crews_thread_safety(self, mock_execute_task, crew_factory):
        mock_execute_task.return_value = "Task completed"
        num_crews = 5

        async def run_crew_async(crew_id: str) -> Dict[str, Any]:
            task_context = {"crew_id": crew_id, "context": None}

            def capture_context(output):
                ctx = get_crew_context()
                task_context["context"] = {
                    "crew_id": ctx.id if ctx else None,
                    "crew_key": ctx.key if ctx else None,
                }
                return output

            crew = crew_factory(crew_id, task_callback=capture_context)
            output = await crew.kickoff_async()

            return {
                "crew_id": crew_id,
                "crew_uuid": str(crew.id),
                "output": output.raw,
                "task_context": task_context,
            }

        tasks = [run_crew_async(f"async_crew_{i}") for i in range(num_crews)]
        results = await asyncio.gather(*tasks)

        for result in results:
            crew_uuid = result["crew_uuid"]
            task_ctx = result["task_context"]["context"]

            assert task_ctx is not None, (
                f"Context should exist during task for {result['crew_id']}"
            )
            assert task_ctx["crew_id"] == crew_uuid, (
                f"Context mismatch for {result['crew_id']}"
            )

    @patch("crewai.Agent.execute_task")
    def test_concurrent_kickoff_for_each(self, mock_execute_task, crew_factory):
        mock_execute_task.return_value = "Task completed"
        contexts_captured = []

        def capture_context(output):
            ctx = get_crew_context()
            contexts_captured.append(
                {
                    "context_id": ctx.id if ctx else None,
                    "thread": threading.current_thread().name,
                }
            )
            return output

        crew = crew_factory("for_each_test", task_callback=capture_context)
        inputs = [{"item": f"input_{i}"} for i in range(3)]

        results = crew.kickoff_for_each(inputs=inputs)

        assert len(results) == len(inputs)
        assert len(contexts_captured) == len(inputs)

        context_ids = [ctx["context_id"] for ctx in contexts_captured]
        assert len(set(context_ids)) == len(inputs), (
            "Each execution should have unique context"
        )

    @patch("crewai.Agent.execute_task")
    def test_no_context_leakage_between_crews(self, mock_execute_task, crew_factory):
        mock_execute_task.return_value = "Task completed"
        contexts = []

        def check_context(output):
            ctx = get_crew_context()
            contexts.append(
                {
                    "context_id": ctx.id if ctx else None,
                    "context_key": ctx.key if ctx else None,
                }
            )
            return output

        def run_crew(name: str):
            crew = crew_factory(name, task_callback=check_context)
            crew.kickoff()
            return str(crew.id)

        crew1_id = run_crew("First")
        crew2_id = run_crew("Second")

        assert len(contexts) == 2
        assert contexts[0]["context_id"] == crew1_id
        assert contexts[1]["context_id"] == crew2_id
        assert contexts[0]["context_id"] != contexts[1]["context_id"]
