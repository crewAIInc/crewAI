from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    FlowCreatedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.flow import Flow, and_, human_feedback, listen, or_, router, start
from crewai.flow.async_feedback import HumanFeedbackPending, PendingFeedbackContext
from crewai.flow.flow import FlowState
from crewai.flow.flow_config import flow_config
from crewai.flow.flow_definition import FlowConfigDefinition, FlowDefinition
from crewai.flow.persistence import persist
from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.runtime._actions import FlowScriptExecutionDisabledError
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.tools import BaseTool
from crewai.types.streaming import FlowStreamingOutput


class StaticSearchTool(BaseTool):
    name: str = "StaticSearchTool"
    description: str = "Returns a deterministic search result."

    def _run(self, search_query: str, prefix: str = "search") -> str:
        return f"{prefix}:{search_query}"


class TypedInputsTool(BaseTool):
    name: str = "TypedInputsTool"
    description: str = "Returns typed input details."

    def _run(self, count: int, include_domains: list[str]) -> str:
        return f"{count}:{','.join(include_domains)}"


class AsyncResultTool(BaseTool):
    name: str = "AsyncResultTool"
    description: str = "Returns an async result from its sync entrypoint."

    def _run(self, value: str) -> Any:
        async def build_result() -> str:
            await asyncio.sleep(0)
            return f"async:{value}"

        return build_result()


class CallableCodeAction:
    def __call__(self, value: str) -> str:
        return f"callable:{value}"


CALLABLE_CODE_ACTION = CallableCodeAction()


class ChainFlow(Flow):
    @start()
    def begin(self):
        self.state["begin_ran"] = True
        return "hello"

    @listen(begin)
    def shout(self, result):
        return result.upper()

    @listen(shout)
    def confirm(self):
        self.state["confirmed"] = True
        return f"confirmed:{self.state['confirmed']}"


class ToolInputFlow(Flow):
    @start()
    def build_query(self):
        self.state["prefix"] = "found"
        return {"query": "ai agents", "suffix": " news"}


class EachActionFlow(Flow):
    inner_thread_id: int | None = None

    def normalize_row(self, row: str, prefix: str = "normalized") -> str:
        return f"{prefix}:{row}"

    def save_row(self, row: str, normalized: str) -> dict[str, str]:
        return {"row": row, "normalized": normalized}

    def keyword_code(self, name: str, punctuation: str) -> str:
        return f"{name}{punctuation}"

    def fail_on_bad_row(self, row: str) -> str:
        if row == "bad":
            raise RuntimeError("bad row")
        return row

    def require_threaded_context(self, row: str) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("each step ran on the event loop")

        from crewai.flow.flow_context import current_flow_method_name

        self.inner_thread_id = threading.get_ident()
        return f"{current_flow_method_name.get()}:{row}"

    def after_each(self) -> str:
        self.state["after_count"] = self.state.get("after_count", 0) + 1
        return f"after:{self.state['after_count']}"


CHAIN_YAML = f"""
schema: crewai.flow/v1
name: ChainFlow
methods:
  begin:
    do:
      call: code
      ref: {__name__}:ChainFlow.begin
    start: true
  shout:
    do:
      ref: {__name__}:ChainFlow.shout
    listen: begin
  confirm:
    do:
      ref: {__name__}:ChainFlow.confirm
    listen: shout
"""


class MergeFlow(Flow):
    @start()
    def begin(self):
        return "go"

    @listen(begin)
    def left(self):
        return "left"

    @listen(begin)
    def right(self):
        return "right"

    @listen(or_(left, right))
    def either(self):
        self.state["either_ran"] = True
        return "either"

    @listen(and_(left, right, either))
    def join(self):
        self.state["joined"] = True
        return "joined"


MERGE_YAML = f"""
schema: crewai.flow/v1
name: MergeFlow
methods:
  begin:
    do:
      ref: {__name__}:MergeFlow.begin
    start: true
  left:
    do:
      ref: {__name__}:MergeFlow.left
    listen: begin
  right:
    do:
      ref: {__name__}:MergeFlow.right
    listen: begin
  either:
    do:
      ref: {__name__}:MergeFlow.either
    listen:
      or: [left, right]
  join:
    do:
      ref: {__name__}:MergeFlow.join
    listen:
      and: [left, right, either]
"""


class RouteFlow(Flow):
    @start()
    def begin(self):
        return "go"

    @router(begin)
    def decide(self):
        return "left" if self.state.get("direction") == "left" else "right"

    @listen("left")
    def take_left(self):
        return "took-left"

    @listen("right")
    def take_right(self):
        return "took-right"


ROUTE_YAML = f"""
schema: crewai.flow/v1
name: RouteFlow
methods:
  begin:
    do:
      ref: {__name__}:RouteFlow.begin
    start: true
  decide:
    do:
      ref: {__name__}:RouteFlow.decide
    listen: begin
    router: true
  take_left:
    do:
      ref: {__name__}:RouteFlow.take_left
    listen: left
  take_right:
    do:
      ref: {__name__}:RouteFlow.take_right
    listen: right
"""


class LoopFlow(Flow):
    @start("retry")
    def step(self):
        self.state["count"] = self.state.get("count", 0) + 1
        return self.state["count"]

    @router(step)
    def decide(self):
        if self.state["count"] < 3:
            return "retry"
        return "done"

    @listen("done")
    def finish(self):
        return f"finished:{self.state['count']}"


LOOP_YAML = f"""
schema: crewai.flow/v1
name: LoopFlow
methods:
  step:
    do:
      ref: {__name__}:LoopFlow.step
    start: retry
  decide:
    do:
      ref: {__name__}:LoopFlow.decide
    listen: step
    router: true
  finish:
    do:
      ref: {__name__}:LoopFlow.finish
    listen: done
"""


class CounterState(FlowState):
    count: int = 0
    label: str = "none"


class PydanticStateFlow(Flow[CounterState]):
    @start()
    def begin(self):
        self.state.count += 1
        return self.state.count

    @listen(begin)
    def finish(self):
        self.state.label = f"count={self.state.count}"
        return self.state.label


PYDANTIC_STATE_YAML = f"""
schema: crewai.flow/v1
name: PydanticStateFlow
state:
  type: pydantic
  ref: {__name__}:CounterState
methods:
  begin:
    do:
      ref: {__name__}:PydanticStateFlow.begin
    start: true
  finish:
    do:
      ref: {__name__}:PydanticStateFlow.finish
    listen: begin
"""

PYDANTIC_STATE_OVERLAY_YAML = f"""
schema: crewai.flow/v1
name: PydanticStateFlow
state:
  type: pydantic
  ref: {__name__}:CounterState
  default:
    count: 5
methods:
  begin:
    do:
      ref: {__name__}:PydanticStateFlow.begin
    start: true
  finish:
    do:
      ref: {__name__}:PydanticStateFlow.finish
    listen: begin
"""

JSON_SCHEMA_STATE_YAML = f"""
schema: crewai.flow/v1
name: JsonSchemaStateFlow
state:
  type: json_schema
  json_schema:
    title: CounterState
    type: object
    properties:
      count:
        type: integer
        default: 0
      label:
        type: string
        default: none
methods:
  begin:
    do:
      ref: {__name__}:PydanticStateFlow.begin
    start: true
  finish:
    do:
      ref: {__name__}:PydanticStateFlow.finish
    listen: begin
"""

PYDANTIC_REF_WITH_SCHEMA_FALLBACK_YAML = f"""
schema: crewai.flow/v1
name: SchemaFallbackFlow
state:
  type: pydantic
  ref: definitely_not_a_module_xyz:MissingState
  json_schema:
    title: CounterState
    type: object
    properties:
      count:
        type: integer
        default: 0
      label:
        type: string
        default: none
methods:
  begin:
    do:
      ref: {__name__}:PydanticStateFlow.begin
    start: true
  finish:
    do:
      ref: {__name__}:PydanticStateFlow.finish
    listen: begin
"""

UNRESOLVABLE_STATE_YAML = f"""
schema: crewai.flow/v1
name: UnresolvableStateFlow
state:
  type: pydantic
  ref: definitely_not_a_module_xyz:MissingState
methods:
  begin:
    do:
      ref: {__name__}:ChainFlow.begin
    start: true
"""

DICT_STATE_YAML = f"""
schema: crewai.flow/v1
name: DictStateFlow
state:
  type: dict
  default:
    count: 5
methods:
  begin:
    do:
      ref: {__name__}:ChainFlow.begin
    start: true
"""

UNKNOWN_STATE_YAML = f"""
schema: crewai.flow/v1
name: UnknownStateFlow
state:
  type: unknown
  ref: somewhere:Something
methods:
  begin:
    do:
      ref: {__name__}:ChainFlow.begin
    start: true
"""


def _run_with_events(flow, inputs=None):
    events = []
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def on_started(source, event):
            events.append(event)

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_finished(source, event):
            events.append(event)

        result = flow.kickoff(inputs=inputs)
    events.sort(key=lambda e: e.timestamp)
    return result, [
        (type(e).__name__, str(e.method_name), e.flow_name) for e in events
    ]


def _state_without_id(flow):
    snapshot = dict(flow.state.model_dump())
    snapshot.pop("id", None)
    return snapshot


def assert_parity(flow_cls, yaml_str, inputs=None, ordered=True):
    class_flow = flow_cls()
    class_result, class_events = _run_with_events(class_flow, inputs)

    definition = FlowDefinition.from_yaml(yaml_str)
    definition_flow = Flow.from_definition(definition)
    definition_result, definition_events = _run_with_events(definition_flow, inputs)

    assert definition_result == class_result
    assert _state_without_id(definition_flow) == _state_without_id(class_flow)
    if ordered:
        assert definition_flow.method_outputs == class_flow.method_outputs
        assert definition_events == class_events
    else:
        assert sorted(map(repr, definition_flow.method_outputs)) == sorted(
            map(repr, class_flow.method_outputs)
        )
        assert sorted(definition_events) == sorted(class_events)
    return definition_flow, definition_result


def test_simple_chain_parity():
    flow, result = assert_parity(ChainFlow, CHAIN_YAML)
    assert result == "confirmed:True"
    assert flow.method_outputs == ["hello", "HELLO", "confirmed:True"]


def test_and_or_merge_parity():
    flow, _ = assert_parity(MergeFlow, MERGE_YAML, ordered=False)
    assert flow.state["joined"] is True
    assert flow.state["either_ran"] is True


def test_router_label_parity_for_each_branch():
    left_flow, _ = assert_parity(RouteFlow, ROUTE_YAML, inputs={"direction": "left"})
    assert "took-left" in left_flow.method_outputs
    assert "took-right" not in left_flow.method_outputs

    right_flow, _ = assert_parity(RouteFlow, ROUTE_YAML, inputs={"direction": "right"})
    assert "took-right" in right_flow.method_outputs


def test_cyclic_flow_parity():
    flow, result = assert_parity(LoopFlow, LOOP_YAML)
    assert result == "finished:3"
    assert flow.state["count"] == 3


def test_definition_flow_events_use_definition_name():
    definition = FlowDefinition.from_yaml(CHAIN_YAML)
    flow = Flow.from_definition(definition)
    _, events = _run_with_events(flow)
    assert events
    assert all(flow_name == "ChainFlow" for _, _, flow_name in events)


def test_definition_method_without_action_is_invalid():
    with pytest.raises(ValidationError, match="do"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "NoActions",
                "methods": {"begin": {"start": True}},
            }
        )


def test_from_definition_unresolvable_ref_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "BadRefs",
            "methods": {
                "begin": {
                    "start": True,
                    "do": {"ref": "definitely_not_a_module_xyz:nope"},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="unresolvable actions.*begin"):
        Flow.from_definition(definition)


def test_from_definition_malformed_ref_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "MalformedRefs",
            "methods": {"begin": {"start": True, "do": {"ref": "no-colon-here"}}},
        }
    )

    with pytest.raises(ValueError, match="expected 'module:qualname'"):
        Flow.from_definition(definition)


def test_from_definition_local_scope_ref_raises():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "LocalRefs",
            "methods": {
                "begin": {
                    "start": True,
                    "do": {"ref": f"{__name__}:make.<locals>.LocalFlow.begin"},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="expected 'module:qualname'"):
        Flow.from_definition(definition)


def test_flow_definition_stamps_refs():
    definition = ChainFlow.flow_definition()

    assert definition.methods["begin"].do.ref == f"{__name__}:ChainFlow.begin"
    assert definition.methods["shout"].do.ref == f"{__name__}:ChainFlow.shout"


def test_from_definition_runs_tool_action_with_static_inputs():
    yaml_str = f"""
schema: crewai.flow/v1
name: ToolFlow
methods:
  search:
    do:
      call: tool
      ref: {__name__}:StaticSearchTool
      with:
        search_query: ai agents
        prefix: found
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff() == "found:ai agents"


def test_tool_action_round_trips_with_inputs():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "ToolFlow",
            "methods": {
                "search": {
                    "start": True,
                    "do": {
                        "call": "tool",
                        "ref": f"{__name__}:StaticSearchTool",
                        "with": {"search_query": "ai agents"},
                    },
                }
            },
        }
    )

    assert definition.to_dict()["methods"]["search"]["do"] == {
        "call": "tool",
        "ref": f"{__name__}:StaticSearchTool",
        "with": {"search_query": "ai agents"},
    }
    assert Flow.from_definition(definition).kickoff() == "search:ai agents"


def test_tool_action_renders_cel_inputs_at_runtime():
    yaml_str = f"""
schema: crewai.flow/v1
name: ToolFlow
methods:
  begin:
    do:
      call: code
      ref: {__name__}:ChainFlow.begin
    start: true
  search:
    do:
      call: tool
      ref: {__name__}:StaticSearchTool
      with:
        search_query: "${{state.begin_ran ? state.topic + ' agents' : 'missing'}}"
        prefix: found
    listen: begin
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"topic": "ai"}) == "found:ai agents"


def test_tool_action_treats_embedded_cel_marker_as_literal():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "ToolFlow",
            "methods": {
                "search": {
                    "start": True,
                    "do": {
                        "call": "tool",
                        "ref": f"{__name__}:StaticSearchTool",
                        "with": {
                            "search_query": "wrapped ${'a}b'} value",
                            "prefix": "${'p}x'}",
                        },
                    },
                },
            },
        }
    )

    assert Flow.from_definition(definition).kickoff() == "p}x:wrapped ${'a}b'} value"


def test_tool_action_treats_marker_with_trailing_text_as_literal():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "ToolFlow",
            "methods": {
                "search": {
                    "start": True,
                    "do": {
                        "call": "tool",
                        "ref": f"{__name__}:StaticSearchTool",
                        "with": {
                            "search_query": "${state.topic} extra",
                            "prefix": "p",
                        },
                    },
                },
            },
        }
    )

    assert Flow.from_definition(definition).kickoff() == "p:${state.topic} extra"


def test_tool_action_rejects_adjacent_markers_as_invalid_cel():
    with pytest.raises(ValidationError, match="invalid CEL expression"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "ToolFlow",
                "methods": {
                    "search": {
                        "start": True,
                        "do": {
                            "call": "tool",
                            "ref": f"{__name__}:StaticSearchTool",
                            "with": {
                                "search_query": "${'a'}${'b'}",
                                "prefix": "p",
                            },
                        },
                    },
                },
            }
        )


def test_tool_action_accepts_braces_in_full_cel_marker():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "ToolFlow",
            "methods": {
                "search": {
                    "start": True,
                    "do": {
                        "call": "tool",
                        "ref": f"{__name__}:StaticSearchTool",
                        "with": {
                            "search_query": "${{'query': 'ai agents'}.query}",
                            "prefix": "${'p}x'}",
                        },
                    },
                },
            },
        }
    )

    assert Flow.from_definition(definition).kickoff() == "p}x:ai agents"


def test_tool_action_renders_latest_output_by_method_name():
    yaml_str = f"""
schema: crewai.flow/v1
name: ToolFlow
methods:
  begin:
    do:
      call: code
      ref: {__name__}:ChainFlow.begin
    start: true
  search:
    do:
      call: tool
      ref: {__name__}:StaticSearchTool
      with:
        search_query: "${{outputs.begin + ' agents'}}"
    listen: begin
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff() == "search:hello agents"


def test_tool_action_uses_state_and_outputs_in_full_yaml_example():
    yaml_str = f"""
schema: crewai.flow/v1
name: ToolFlow
methods:
  build_query:
    do:
      call: code
      ref: {__name__}:ToolInputFlow.build_query
    start: true
  search:
    do:
      call: tool
      ref: {__name__}:StaticSearchTool
      with:
        search_query: "${{outputs.build_query.query + outputs.build_query.suffix}}"
        prefix: "${{state.prefix}}"
    listen: build_query
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff() == "found:ai agents news"


def test_tool_action_preserves_whole_expression_value_types():
    yaml_str = f"""
schema: crewai.flow/v1
name: ToolFlow
methods:
  typed:
    do:
      call: tool
      ref: {__name__}:TypedInputsTool
      with:
        count: "${{state.limit}}"
        include_domains: "${{state.domains}}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert (
        flow.kickoff(inputs={"limit": 2, "domains": ["crewai.com", "example.com"]})
        == "2:crewai.com,example.com"
    )


def test_agent_action_runs_inline_yaml_definition(monkeypatch: pytest.MonkeyPatch):
    from crewai import Agent

    async def fake_kickoff_async(
        self: Agent, messages: str, **_kwargs: Any
    ) -> dict[str, Any]:
        return {"agent": self.role, "input": messages}

    monkeypatch.setattr(Agent, "kickoff_async", fake_kickoff_async)

    yaml_str = """
schema: crewai.flow/v1
name: AgentFlow
methods:
  answer:
    do:
      call: agent
      with:
        role: Analyst
        goal: Answer questions
        backstory: Knows things.
        input: "${state.question}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"question": "What is CrewAI?"}) == {
        "agent": "Analyst",
        "input": "What is CrewAI?",
    }


def test_agent_action_runs_inside_each(monkeypatch: pytest.MonkeyPatch):
    from crewai import Agent

    async def fake_kickoff_async(
        self: Agent, messages: str, **_kwargs: Any
    ) -> str:
        return f"{self.role}:{messages}"

    monkeypatch.setattr(Agent, "kickoff_async", fake_kickoff_async)

    yaml_str = """
schema: crewai.flow/v1
name: AgentEachFlow
methods:
  answer_each:
    do:
      call: each
      in: state.questions
      do:
        - name: answer
          action:
            call: agent
            with:
              role: Analyst
              goal: Answer questions
              backstory: Knows things.
              input: "${item}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"questions": ["one", "two"]}) == [
        "Analyst:one",
        "Analyst:two",
    ]


def test_agent_action_round_trips_with_inline_definition():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "AgentFlow",
            "methods": {
                "answer": {
                    "start": True,
                    "do": {
                        "call": "agent",
                        "with": {
                            "role": "Analyst",
                            "goal": "Answer questions",
                            "backstory": "Knows things.",
                            "settings": {"verbose": True},
                            "input": "${state.question}",
                        },
                    },
                }
            },
        }
    )

    round_trip = FlowDefinition.from_yaml(definition.to_yaml())
    action = round_trip.to_dict()["methods"]["answer"]["do"]

    assert action["call"] == "agent"
    assert action["with"]["role"] == "Analyst"
    assert action["with"]["input"] == "${state.question}"
    assert action["with"]["settings"] == {"verbose": True}


def test_agent_action_json_schema_describes_inline_agent_definitions():
    schema_defs = FlowDefinition.json_schema()["$defs"]

    assert set(schema_defs["AgentDefinition"]["properties"]) >= {
        "role",
        "goal",
        "backstory",
        "settings",
        "input",
        "response_format",
    }


def test_agent_action_rejects_non_string_input_in_definition():
    with pytest.raises(ValidationError, match="agent.input must be a string"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "AgentFlow",
                "methods": {
                    "answer": {
                        "start": True,
                        "do": {
                            "call": "agent",
                            "with": {
                                "role": "Analyst",
                                "goal": "Answer questions",
                                "backstory": "Knows things.",
                                "input": 123,
                            },
                        },
                    }
                },
            }
        )


def test_agent_action_reports_invalid_cel_expression():
    yaml_str = """
schema: crewai.flow/v1
name: AgentFlow
methods:
  answer:
    do:
      call: agent
      with:
        role: Analyst
        goal: Answer questions
        backstory: Knows things.
        input: "${state.}"
    start: true
"""

    with pytest.raises(ValidationError, match="invalid CEL expression"):
        FlowDefinition.from_yaml(yaml_str)


def test_crew_action_runs_inline_yaml_definition(monkeypatch: pytest.MonkeyPatch):
    from crewai import Crew

    async def fake_kickoff_async(
        self: Crew, inputs: dict[str, Any] | None = None, **_kwargs: Any
    ) -> dict[str, Any]:
        return {
            "crew": self.name,
            "agents": [agent.role for agent in self.agents],
            "tasks": [task.description for task in self.tasks],
            "inputs": inputs,
        }

    monkeypatch.setattr(Crew, "kickoff_async", fake_kickoff_async)

    yaml_str = """
schema: crewai.flow/v1
name: CrewFlow
methods:
  research:
    do:
      call: crew
      with:
        name: inline_research
        agents:
          researcher:
            role: Researcher
            goal: Research {topic}
            backstory: Knows things.
        tasks:
          - name: research_task
            description: Research {topic}
            expected_output: Findings about {topic}
            agent: researcher
      inputs:
        topic: "${state.topic}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"topic": "AI"}) == {
        "crew": "inline_research",
        "agents": ["Researcher"],
        "tasks": ["Research {topic}"],
        "inputs": {"topic": "AI"},
    }


def test_crew_action_runs_crew_from_declaration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from crewai import Crew

    project_root = tmp_path / "project"
    crew_root = project_root / "crews" / "research_crew"
    agents_root = crew_root / "agents"
    agents_root.mkdir(parents=True)
    (agents_root / "researcher.jsonc").write_text(
        """
{
  "role": "Researcher",
  "goal": "Research {topic}",
  "backstory": "Knows things."
}
""",
        encoding="utf-8",
    )
    (crew_root / "crew.jsonc").write_text(
        """
{
  "name": "referenced_research",
  "agents": ["researcher"],
  "tasks": [
    {
      "name": "research_task",
      "description": "Research {topic}",
      "expected_output": "Findings about {topic}",
      "agent": "researcher"
    }
  ],
  "inputs": {
    "topic": "Default topic",
    "audience": "developers"
  }
}
""",
        encoding="utf-8",
    )

    async def fake_kickoff_async(
        self: Crew, inputs: dict[str, Any] | None = None, **_kwargs: Any
    ) -> dict[str, Any]:
        return {
            "crew": self.name,
            "tasks": [task.description for task in self.tasks],
            "inputs": inputs,
        }

    monkeypatch.setattr(Crew, "kickoff_async", fake_kickoff_async)
    monkeypatch.chdir(project_root)

    yaml_str = """
schema: crewai.flow/v1
name: CrewFlow
methods:
  research:
    do:
      call: crew
      from_declaration: crews/research_crew
      inputs:
        topic: "${state.topic}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"topic": "AI"}) == {
        "crew": "referenced_research",
        "tasks": ["Research {topic}"],
        "inputs": {"topic": "AI", "audience": "developers"},
    }


def test_crew_action_from_declaration_resolves_relative_to_flow_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from crewai import Crew

    project_root = tmp_path / "project"
    crew_root = project_root / "crews" / "research_crew"
    agents_root = crew_root / "agents"
    agents_root.mkdir(parents=True)
    (agents_root / "researcher.jsonc").write_text(
        """
{
  "role": "Researcher",
  "goal": "Research {topic}",
  "backstory": "Knows things."
}
""",
        encoding="utf-8",
    )
    (crew_root / "crew.jsonc").write_text(
        """
{
  "name": "relative_research",
  "agents": ["researcher"],
  "tasks": [
    {
      "description": "Research {topic}",
      "expected_output": "Findings about {topic}",
      "agent": "researcher"
    }
  ],
  "inputs": {
    "topic": "Default topic"
  }
}
""",
        encoding="utf-8",
    )

    async def fake_kickoff_async(
        self: Crew, inputs: dict[str, Any] | None = None, **_kwargs: Any
    ) -> dict[str, Any]:
        return {"crew": self.name, "inputs": inputs}

    monkeypatch.setattr(Crew, "kickoff_async", fake_kickoff_async)

    flow_path = project_root / "flow.yaml"
    yaml_str = """
schema: crewai.flow/v1
name: CrewFlow
methods:
  research:
    do:
      call: crew
      from_declaration: crews/research_crew
      inputs:
        topic: "${state.topic}"
    start: true
"""
    flow_path.write_text(yaml_str, encoding="utf-8")

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    flow = Flow.from_definition(
        FlowDefinition.from_yaml(yaml_str, source_path=flow_path)
    )

    assert flow.kickoff(inputs={"topic": "AI"}) == {
        "crew": "relative_research",
        "inputs": {"topic": "AI"},
    }


def test_crew_action_from_declaration_rejects_paths_outside_flow_file(
    tmp_path: Path,
):
    flow_path = tmp_path / "project" / "flow.yaml"
    flow_path.parent.mkdir()
    yaml_str = """
schema: crewai.flow/v1
name: CrewFlow
methods:
  research:
    do:
      call: crew
      from_declaration: ../outside/crew.jsonc
    start: true
"""

    flow = Flow.from_definition(
        FlowDefinition.from_yaml(yaml_str, source_path=flow_path)
    )

    with pytest.raises(
        ValueError,
        match="crew declaration path must be within the flow definition directory",
    ):
        flow.kickoff()


def test_crew_action_round_trips_with_inline_definition():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "CrewFlow",
            "methods": {
                "research": {
                    "start": True,
                    "do": {
                        "call": "crew",
                        "with": {
                            "name": "inline_research",
                            "agents": {
                                "researcher": {
                                    "role": "Researcher",
                                    "goal": "Research {topic}",
                                    "backstory": "Knows things.",
                                }
                            },
                            "tasks": [
                                {
                                    "name": "research_task",
                                    "description": "Research {topic}",
                                    "expected_output": "Findings about {topic}",
                                    "agent": "researcher",
                                }
                            ],
                        },
                        "inputs": {"topic": "${state.topic}"},
                    },
                }
            },
        }
    )

    assert definition.to_dict()["methods"]["research"]["do"]["call"] == "crew"
    assert (
        definition.to_dict()["methods"]["research"]["do"]["with"]["agents"][
            "researcher"
        ]["role"]
        == "Researcher"
    )
    assert definition.to_dict()["methods"]["research"]["do"]["inputs"] == {
        "topic": "${state.topic}"
    }


def test_crew_action_normalizes_named_agent_list_definition():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "CrewFlow",
            "methods": {
                "research": {
                    "start": True,
                    "do": {
                        "call": "crew",
                        "with": {
                            "agents": [
                                {
                                    "name": "researcher",
                                    "role": "Researcher",
                                    "goal": "Research {topic}",
                                    "backstory": "Knows things.",
                                }
                            ],
                            "tasks": [
                                {
                                    "description": "Research {topic}",
                                    "expected_output": "Findings about {topic}",
                                    "agent": "researcher",
                                }
                            ],
                        },
                    },
                }
            },
        }
    )

    assert (
        definition.to_dict()["methods"]["research"]["do"]["with"]["agents"][
            "researcher"
        ]["role"]
        == "Researcher"
    )


def test_crew_action_json_schema_describes_inline_crew_definitions():
    schema_defs = FlowDefinition.json_schema()["$defs"]
    agents_schema = schema_defs["CrewDefinition"]["properties"]["agents"]

    assert set(schema_defs["CrewDefinition"]["properties"]) >= {
        "agents",
        "tasks",
        "inputs",
    }
    assert {option["type"] for option in agents_schema["anyOf"]} == {"array", "object"}
    assert set(schema_defs["CrewAgentDefinition"]["properties"]) >= {
        "role",
        "goal",
        "backstory",
        "settings",
    }
    assert set(schema_defs["CrewTaskDefinition"]["properties"]) >= {
        "description",
        "expected_output",
        "agent",
        "context",
    }


def test_crew_action_rejects_incomplete_inline_agent_definition():
    with pytest.raises(ValidationError, match="goal"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "CrewFlow",
                "methods": {
                    "research": {
                        "start": True,
                        "do": {
                            "call": "crew",
                            "with": {
                                "agents": {
                                    "researcher": {
                                        "role": "Researcher",
                                        "backstory": "Knows things.",
                                    }
                                },
                                "tasks": [
                                    {
                                        "description": "Research",
                                        "expected_output": "Findings",
                                        "agent": "researcher",
                                    }
                                ],
                            },
                        },
                    }
                },
            }
        )


def test_crew_action_rejects_python_ref_field():
    with pytest.raises(ValidationError, match="ref"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "CrewFlow",
                "methods": {
                    "research": {
                        "start": True,
                        "do": {
                            "call": "crew",
                            "ref": "project.crew:build_crew",
                        },
                    }
                },
            }
        )


def test_crew_action_rejects_non_mapping_inputs_in_definition():
    with pytest.raises(ValidationError, match="crew.inputs must be a mapping"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "CrewFlow",
                "methods": {
                    "research": {
                        "start": True,
                        "do": {
                            "call": "crew",
                            "with": {
                                "agents": {
                                    "researcher": {
                                        "role": "Researcher",
                                        "goal": "Research",
                                        "backstory": "Knows things.",
                                    }
                                },
                                "tasks": [
                                    {
                                        "description": "Research",
                                        "expected_output": "Findings",
                                        "agent": "researcher",
                                    }
                                ],
                                "inputs": "topic",
                            },
                        },
                    }
                },
            }
        )


def test_tool_action_reports_invalid_cel_expression():
    yaml_str = f"""
schema: crewai.flow/v1
name: ToolFlow
methods:
  search:
    do:
      call: tool
      ref: {__name__}:StaticSearchTool
      with:
        search_query: "${{state.}}"
    start: true
"""

    with pytest.raises(ValidationError, match="invalid CEL expression"):
        FlowDefinition.from_yaml(yaml_str)


def test_code_action_renders_keyword_inputs():
    yaml_str = f"""
schema: crewai.flow/v1
name: CodeWithFlow
methods:
  greet:
    do:
      call: code
      ref: {__name__}:EachActionFlow.keyword_code
      with:
        name: "${{state.name}}"
        punctuation: "!"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"name": "hello"}) == "hello!"


def test_code_action_supports_callable_instance_refs():
    yaml_str = f"""
schema: crewai.flow/v1
name: CallableInstanceFlow
methods:
  call_instance:
    do:
      call: code
      ref: {__name__}:CALLABLE_CODE_ACTION
      with:
        value: "${{state.value}}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"value": "ok"}) == "callable:ok"


def test_each_action_executes_one_nested_code_action():
    yaml_str = f"""
schema: crewai.flow/v1
name: EachFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: normalize
          action:
            call: code
            ref: {__name__}:EachActionFlow.normalize_row
            with:
              row: "${{item}}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"rows": ["a", "b"]}) == [
        "normalized:a",
        "normalized:b",
    ]


def test_each_action_runs_sync_steps_off_event_loop_with_context():
    yaml_str = f"""
schema: crewai.flow/v1
name: EachFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: threaded
          action:
            call: code
            ref: {__name__}:EachActionFlow.require_threaded_context
            with:
              row: "${{item}}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    caller_thread_id = threading.get_ident()

    assert flow.kickoff(inputs={"rows": ["a"]}) == ["process_rows:a"]
    assert flow.inner_thread_id is not None
    assert flow.inner_thread_id != caller_thread_id


def test_each_action_runs_async_tool_results_from_sync_steps():
    yaml_str = f"""
schema: crewai.flow/v1
name: EachFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: async_tool
          action:
            call: tool
            ref: {__name__}:AsyncResultTool
            with:
              value: "${{item}}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"rows": ["a", "b"]}) == ["async:a", "async:b"]


def test_script_action_requires_explicit_opt_in():
    yaml_str = """
schema: crewai.flow/v1
name: ScriptFlow
methods:
  normalize:
    do:
      call: script
      code: |
        return "blocked"
    start: true
"""

    with pytest.raises(
        FlowScriptExecutionDisabledError,
        match="CREWAI_ALLOW_FLOW_SCRIPT_EXECUTION=1",
    ) as exc_info:
        Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    assert "methods with unresolvable actions" not in str(exc_info.value)


def test_script_action_runs_python_imports_mutates_state_and_returns_value(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CREWAI_ALLOW_FLOW_SCRIPT_EXECUTION", "1")

    yaml_str = """
schema: crewai.flow/v1
name: ScriptFlow
methods:
  normalize:
    do:
      call: script
      code: |
        import math

        state["rounded"] = math.ceil(state["raw_score"])
        return f"rounded:{state['rounded']}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"raw_score": 3.2}) == "rounded:4"
    assert flow.state["rounded"] == 4


def test_script_listener_reads_trigger_input_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CREWAI_ALLOW_FLOW_SCRIPT_EXECUTION", "1")

    yaml_str = """
schema: crewai.flow/v1
name: ScriptFlow
methods:
  seed:
    do:
      call: expression
      expr: "'alpha'"
    start: true
  combine:
    do:
      call: script
      code: |
        state["input_matches_output"] = input == outputs["seed"]
        return f"{outputs['seed']}:{input}"
    listen: seed
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff() == "alpha:alpha"
    assert flow.state["input_matches_output"] is True


def test_script_each_action_reads_item_and_step_outputs(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CREWAI_ALLOW_FLOW_SCRIPT_EXECUTION", "1")

    yaml_str = """
schema: crewai.flow/v1
name: ScriptEachFlow
methods:
  seed:
    do:
      call: expression
      expr: "'global'"
    start: true
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: clean
          action:
            call: script
            code: |
              return item.strip()
        - name: tag
          action:
            call: script
            code: |
              return f"{outputs['seed']}:{outputs['clean']}"
    listen: seed
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"rows": [" a ", " b "]}) == ["global:a", "global:b"]


def test_each_action_uses_iteration_outputs_between_steps():
    yaml_str = f"""
schema: crewai.flow/v1
name: EachFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: normalize
          action:
            call: code
            ref: {__name__}:EachActionFlow.normalize_row
            with:
              row: "${{item}}"
              prefix: saved
        - name: save
          action:
            call: code
            ref: {__name__}:EachActionFlow.save_row
            with:
              row: "${{item}}"
              normalized: "${{outputs.normalize}}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"rows": ["a", "b"]}) == [
        {"row": "a", "normalized": "saved:a"},
        {"row": "b", "normalized": "saved:b"},
    ]


def test_each_action_resets_step_outputs_between_iterations():
    yaml_str = """
schema: crewai.flow/v1
name: EachFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: leak_check
          action:
            call: expression
            expr: "has(outputs.previous) ? outputs.previous : 'empty'"
        - name: previous
          action:
            call: expression
            expr: item
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"rows": ["a", "b"]}) == ["a", "b"]
    assert flow._method_outputs == [
        {"method": "process_rows", "output": ["a", "b"]}
    ]


def test_each_action_preserves_flow_outputs_and_prefers_step_outputs():
    yaml_str = """
schema: crewai.flow/v1
name: EachFlow
methods:
  seed:
    do:
      call: expression
      expr: "'global'"
    start: true
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: before_shadow
          action:
            call: expression
            expr: "outputs.seed + ':' + item"
        - name: seed
          action:
            call: expression
            expr: "'local:' + item"
        - name: after_shadow
          action:
            call: expression
            expr: "outputs.seed"
    listen: seed
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"rows": ["a", "b"]}) == [
        "local:a",
        "local:b",
    ]
    assert flow._method_outputs == [
        {"method": "seed", "output": "global"},
        {"method": "process_rows", "output": ["local:a", "local:b"]},
    ]


def test_each_action_runs_simple_if_clauses():
    yaml_str = """
schema: crewai.flow/v1
name: EachIfFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: kind
          action:
            call: expression
            expr: item.kind
        - name: kept
          if: "outputs.kind == 'keep'"
          action:
            call: expression
            expr: "'kept:' + item.value"
        - name: skipped
          if: "outputs.kind != 'keep'"
          action:
            call: expression
            expr: "'skipped:' + item.value"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(
        inputs={
            "rows": [
                {"kind": "keep", "value": "a"},
                {"kind": "drop", "value": "b"},
            ]
        }
    ) == ["kept:a", "skipped:b"]


def test_each_action_accepts_expression_markers_in_explicit_cel_fields():
    yaml_str = """
schema: crewai.flow/v1
name: EachIfFlow
methods:
  process_rows:
    do:
      call: each
      in: "${state.rows}"
      do:
        - name: kind
          action:
            call: expression
            expr: "${item.kind}"
        - name: kept
          if: "${outputs.kind == 'keep'}"
          action:
            call: expression
            expr: "${item.value}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(inputs={"rows": [{"kind": "keep", "value": "a"}]}) == ["a"]


def test_each_action_skipped_if_keeps_previous_output():
    yaml_str = """
schema: crewai.flow/v1
name: EachIfFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: original
          action:
            call: expression
            expr: item.value
        - name: maybe_included
          if: item.include
          action:
            call: expression
            expr: "'included:' + item.value"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    assert flow.kickoff(
        inputs={
            "rows": [
                {"include": True, "value": "a"},
                {"include": False, "value": "b"},
            ]
        }
    ) == ["included:a", "b"]


def test_each_action_if_condition_must_be_boolean():
    yaml_str = """
schema: crewai.flow/v1
name: EachIfFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: value
          if: item.value
          action:
            call: expression
            expr: item.value
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    with pytest.raises(ValueError, match="if expression must evaluate to a boolean"):
        flow.kickoff(inputs={"rows": [{"value": "truthy"}]})


def test_each_action_empty_list_returns_empty_and_listener_runs_once():
    yaml_str = f"""
schema: crewai.flow/v1
name: EachFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: normalize
          action:
            call: code
            ref: {__name__}:EachActionFlow.normalize_row
            with:
              row: "${{item}}"
    start: true
  after_each:
    do:
      call: code
      ref: {__name__}:EachActionFlow.after_each
    listen: process_rows
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    events = []
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def on_finished(source, event):
            events.append(event.method_name)

        result = flow.kickoff(inputs={"rows": []})

    assert result == "after:1"
    assert flow.method_outputs == [[], "after:1"]
    assert flow.state["after_count"] == 1
    assert events.count("process_rows") == 1
    assert events.count("after_each") == 1


@pytest.mark.parametrize(
    ("expr", "inputs"),
    [
        ("1", {}),
        ('"rows"', {}),
        ("state.rows", {"rows": {"a": 1}}),
    ],
)
def test_each_action_rejects_non_list_inputs(expr, inputs):
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "EachFlow",
            "methods": {
                "process_rows": {
                    "start": True,
                    "do": {
                        "call": "each",
                        "in": expr,
                        "do": [
                            {
                                "name": "value",
                                "action": {"call": "expression", "expr": "item"},
                            }
                        ],
                    },
                }
            },
        }
    )
    flow = Flow.from_definition(definition)

    with pytest.raises(ValueError, match="each.in must evaluate to an array"):
        flow.kickoff(inputs=inputs)


@pytest.mark.parametrize(
    "action_do",
    [
        [],
        [{"value": {"call": "expression", "expr": "item"}}],
        [{"name": "1bad", "action": {"call": "expression", "expr": "item"}}],
        [{"name": "missing_action"}],
        [{"action": {"call": "expression", "expr": "item"}}],
        [
            {
                "name": "value",
                "if": "true",
                "then": [],
                "action": {"call": "expression", "expr": "item"},
            }
        ],
        [
            {"name": "same", "action": {"call": "expression", "expr": "item"}},
            {"name": "same", "action": {"call": "expression", "expr": "item"}},
        ],
    ],
)
def test_each_action_validates_step_shape(action_do):
    with pytest.raises(ValidationError):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "EachFlow",
                "methods": {
                    "process_rows": {
                        "start": True,
                        "do": {
                            "call": "each",
                            "in": "state.rows",
                            "do": action_do,
                        },
                    }
                },
            }
        )


def test_if_clauses_are_rejected_at_method_level():
    with pytest.raises(ValidationError):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "TopLevelIfFlow",
                "methods": {
                    "process": {
                        "start": True,
                        "do": {
                            "call": "expression",
                            "if": "true",
                            "expr": "'ok'",
                        },
                    }
                },
            }
        )


def test_each_action_rejects_nested_each_actions():
    with pytest.raises(ValidationError):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v1",
                "name": "EachFlow",
                "methods": {
                    "process_rows": {
                        "start": True,
                        "do": {
                            "call": "each",
                            "in": "state.rows",
                            "do": [
                                {
                                    "name": "nested",
                                    "action": {
                                        "call": "each",
                                        "in": "state.children",
                                        "do": [
                                            {
                                                "name": "child",
                                                "action": {
                                                    "call": "expression",
                                                    "expr": "item",
                                                }
                                            }
                                        ],
                                    }
                                }
                            ],
                        },
                    }
                },
            }
        )


def test_each_action_failure_fails_outer_method():
    yaml_str = f"""
schema: crewai.flow/v1
name: EachFlow
methods:
  process_rows:
    do:
      call: each
      in: state.rows
      do:
        - name: validate
          action:
            call: code
            ref: {__name__}:EachActionFlow.fail_on_bad_row
            with:
              row: "${{item}}"
    start: true
"""

    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    with pytest.raises(RuntimeError, match="bad row"):
        flow.kickoff(inputs={"rows": ["ok", "bad"]})


def test_expression_action_round_trips():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "ExpressionFlow",
            "methods": {
                "classify": {
                    "start": True,
                    "do": {
                        "call": "expression",
                        "expr": "state.score >= 80 ? 'qualified' : 'nurture'",
                    },
                }
            },
        }
    )

    assert definition.to_dict()["methods"]["classify"]["do"] == {
        "call": "expression",
        "expr": "state.score >= 80 ? 'qualified' : 'nurture'",
    }
    assert Flow.from_definition(definition).kickoff(inputs={"score": 90}) == "qualified"


def test_explicit_cel_fields_accept_expression_markers():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "ExpressionFlow",
            "methods": {
                "classify": {
                    "start": True,
                    "do": {
                        "call": "expression",
                        "expr": "${state.score >= 80 ? 'qualified' : 'nurture'}",
                    },
                }
            },
        }
    )

    assert Flow.from_definition(definition).kickoff(inputs={"score": 90}) == "qualified"


def test_expression_local_context_recurses_into_dataclass_values():
    from crewai.flow.expressions import Expression

    class Payload(BaseModel):
        name: str

    @dataclass
    class Row:
        payload: Payload

    assert (
        Expression.from_flow(
            "item.payload.name",
            Flow(),
            local_context={"item": Row(payload=Payload(name="qualified"))},
        ).evaluate()
        == "qualified"
    )


def test_expression_empty_context_overrides_stored_context():
    from crewai.flow.expressions import Expression, ExpressionError

    expression = Expression("state.score", context={"state": {"score": 90}})

    assert expression.evaluate() == 90
    with pytest.raises(ExpressionError):
        expression.evaluate({})


def test_expression_template_empty_context_overrides_stored_context():
    from crewai.flow.expressions import Expression, ExpressionError

    expression = Expression(
        {"score": "${state.score}"}, context={"state": {"score": 90}}
    )

    assert expression.render_template() == {"score": 90}
    with pytest.raises(ExpressionError):
        expression.render_template({})


def test_expression_action_can_route_like_if_else():
    yaml_str = f"""
schema: crewai.flow/v1
name: ExpressionRouterFlow
methods:
  begin:
    do:
      call: code
      ref: {__name__}:ChainFlow.begin
    start: true
  decide:
    do:
      call: expression
      expr: "state.direction == 'left' ? 'left' : 'right'"
    listen: begin
    router: true
    emit: [left, right]
  take_left:
    do:
      call: code
      ref: {__name__}:RouteFlow.take_left
    listen: left
  take_right:
    do:
      call: code
      ref: {__name__}:RouteFlow.take_right
    listen: right
"""

    definition = FlowDefinition.from_yaml(yaml_str)

    assert Flow.from_definition(definition).kickoff(
        inputs={"direction": "left"}
    ) == "took-left"
    assert Flow.from_definition(definition).kickoff(
        inputs={"direction": "right"}
    ) == "took-right"


def test_expression_action_reports_invalid_cel_expression():
    yaml_str = """
schema: crewai.flow/v1
name: ExpressionFlow
methods:
  classify:
    do:
      call: expression
      expr: "state."
    start: true
"""

    with pytest.raises(ValidationError, match="invalid CEL expression"):
        FlowDefinition.from_yaml(yaml_str)


def test_expression_action_rejects_unknown_cel_root():
    yaml_str = """
schema: crewai.flow/v1
name: ExpressionFlow
methods:
  classify:
    do:
      call: expression
      expr: "score >= 80"
    start: true
"""

    with pytest.raises(ValidationError, match="unknown CEL root"):
        FlowDefinition.from_yaml(yaml_str)


def test_tool_action_requires_module_qualname_ref():
    definition = FlowDefinition.from_dict(
        {
            "schema": "crewai.flow/v1",
            "name": "ToolFlow",
            "methods": {
                "search": {
                    "start": True,
                    "do": {
                        "call": "tool",
                        "ref": f"{__name__}.StaticSearchTool",
                        "with": {"search_query": "ai agents"},
                    },
                }
            },
        }
    )

    with pytest.raises(ValueError, match="expected 'module:qualname'"):
        Flow.from_definition(definition)


def test_pydantic_state_from_ref_parity():
    flow, result = assert_parity(PydanticStateFlow, PYDANTIC_STATE_YAML)
    assert result == "count=1"
    assert flow.state.count == 1
    assert flow.state.label == "count=1"
    assert flow.state.id


def test_pydantic_state_default_overlay():
    flow = Flow.from_definition(FlowDefinition.from_yaml(PYDANTIC_STATE_OVERLAY_YAML))
    result = flow.kickoff()
    assert result == "count=6"
    assert flow.state.count == 6


def test_json_schema_state():
    flow = Flow.from_definition(FlowDefinition.from_yaml(JSON_SCHEMA_STATE_YAML))
    result = flow.kickoff()
    assert result == "count=1"
    assert flow.state.count == 1
    assert flow.state.label == "count=1"
    assert flow.state.id


def test_json_schema_state_validates_inputs():
    flow = Flow.from_definition(FlowDefinition.from_yaml(JSON_SCHEMA_STATE_YAML))
    with pytest.raises(ValueError, match="Invalid inputs"):
        flow.kickoff(inputs={"count": "not-a-number"})


def test_pydantic_state_falls_back_to_json_schema_when_ref_unimportable():
    flow = Flow.from_definition(
        FlowDefinition.from_yaml(PYDANTIC_REF_WITH_SCHEMA_FALLBACK_YAML)
    )
    result = flow.kickoff()
    assert result == "count=1"
    assert flow.state.count == 1


def test_pydantic_state_without_ref_or_schema_falls_back_to_dict(caplog):
    with caplog.at_level("ERROR"):
        flow = Flow.from_definition(FlowDefinition.from_yaml(UNRESOLVABLE_STATE_YAML))
    assert "falling back to dict state" in caplog.text

    result = flow.kickoff()
    assert result == "hello"
    assert flow.state["begin_ran"] is True
    assert flow.state["id"]


def test_dict_state_is_a_copy_of_default_plus_id():
    definition = FlowDefinition.from_yaml(DICT_STATE_YAML)

    flow = Flow.from_definition(definition)
    assert flow.state["count"] == 5
    assert flow.state["id"]
    flow.kickoff()
    assert flow.state["begin_ran"] is True

    second = Flow.from_definition(definition)
    assert second.state["count"] == 5
    assert "begin_ran" not in second.state
    assert second.state["id"] != flow.state["id"]
    assert definition.state.default == {"count": 5}


def test_unknown_state_type_falls_back_to_dict(caplog):
    with caplog.at_level("WARNING"):
        flow = Flow.from_definition(FlowDefinition.from_yaml(UNKNOWN_STATE_YAML))
    assert "falling back to dict state" in caplog.text

    result = flow.kickoff()
    assert result == "hello"
    assert flow.state["begin_ran"] is True


class StubInputProvider:
    def request_input(self, message, flow, metadata=None):
        return "stub"


class ConfiguredFlow(Flow):
    suppress_flow_events = True
    max_method_calls = 5
    input_provider = StubInputProvider()

    @start()
    def begin(self):
        return "configured"


SUPPRESSED_CHAIN_YAML = (
    CHAIN_YAML
    + """
config:
  suppress_flow_events: true
"""
)

CAPPED_LOOP_YAML = (
    LOOP_YAML
    + """
config:
  max_method_calls: 2
"""
)

STREAMING_CHAIN_YAML = (
    CHAIN_YAML
    + """
config:
  stream: true
"""
)

DEFERRED_CHAIN_YAML = (
    CHAIN_YAML
    + """
config:
  defer_trace_finalization: true
"""
)

INPUT_PROVIDER_CHAIN_YAML = (
    CHAIN_YAML
    + f"""
config:
  input_provider: {__name__}:StubInputProvider
"""
)


def _run_capturing_flow_lifecycle(yaml_str, event_types):
    events = []
    with crewai_event_bus.scoped_handlers():
        for event_type in event_types:

            @crewai_event_bus.on(event_type)
            def capture(source, event):
                events.append(event)

        flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
        result = flow.kickoff()
    return flow, result, events


_LIFECYCLE_EVENTS = [
    FlowCreatedEvent,
    FlowStartedEvent,
    FlowFinishedEvent,
    MethodExecutionStartedEvent,
    MethodExecutionFinishedEvent,
]


def test_config_suppress_flow_events_from_yaml():
    twin_events = []
    with crewai_event_bus.scoped_handlers():
        for event_type in _LIFECYCLE_EVENTS:

            @crewai_event_bus.on(event_type)
            def capture(source, event):
                twin_events.append(type(event).__name__)

        twin_result = ChainFlow(suppress_flow_events=True).kickoff()

    flow, result, events = _run_capturing_flow_lifecycle(
        SUPPRESSED_CHAIN_YAML, _LIFECYCLE_EVENTS
    )
    assert result == twin_result == "confirmed:True"
    assert flow.suppress_flow_events is True
    assert [type(e).__name__ for e in events] == twin_events
    assert not any(
        isinstance(e, (MethodExecutionStartedEvent, MethodExecutionFinishedEvent))
        for e in events
    )


def test_config_max_method_calls_from_yaml():
    flow = Flow.from_definition(FlowDefinition.from_yaml(CAPPED_LOOP_YAML))
    with pytest.raises(RecursionError, match="has been called 2 times"):
        flow.kickoff()


def test_config_stream_from_yaml():
    flow = Flow.from_definition(FlowDefinition.from_yaml(STREAMING_CHAIN_YAML))
    streaming = flow.kickoff()
    assert isinstance(streaming, FlowStreamingOutput)
    for _ in streaming:
        pass
    assert streaming.result == "confirmed:True"
    assert flow.stream is True


def test_config_defer_trace_finalization_from_yaml():
    _, _, baseline_events = _run_capturing_flow_lifecycle(
        CHAIN_YAML, [FlowFinishedEvent]
    )
    assert len(baseline_events) == 1

    flow, result, deferred_events = _run_capturing_flow_lifecycle(
        DEFERRED_CHAIN_YAML, [FlowFinishedEvent]
    )
    assert result == "confirmed:True"
    assert flow.defer_trace_finalization is True
    assert deferred_events == []


def test_config_checkpoint_from_yaml(tmp_path):
    yaml_str = (
        CHAIN_YAML
        + f"""
config:
  checkpoint:
    location: {tmp_path}
"""
    )
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    assert isinstance(flow.checkpoint, CheckpointConfig)
    assert flow.checkpoint.location == str(tmp_path)


def test_config_input_provider_from_yaml():
    flow = Flow.from_definition(FlowDefinition.from_yaml(INPUT_PROVIDER_CHAIN_YAML))
    assert isinstance(flow.input_provider, StubInputProvider)


def test_round_trip_config_equivalence():
    class_flow = ConfiguredFlow()
    definition = FlowDefinition.from_yaml(ConfiguredFlow.flow_definition().to_yaml())
    definition_flow = Flow.from_definition(definition)

    assert definition.config.suppress_flow_events is True
    assert definition.config.max_method_calls == 5
    assert definition.config.input_provider == f"{__name__}:StubInputProvider"
    assert definition_flow.suppress_flow_events is class_flow.suppress_flow_events
    assert definition_flow.max_method_calls == class_flow.max_method_calls
    assert isinstance(definition_flow.input_provider, StubInputProvider)

    class_result, class_events = _run_with_events(class_flow)
    definition_result, definition_events = _run_with_events(definition_flow)
    assert definition_result == class_result == "configured"
    assert definition_events == class_events


def test_unknown_schema_rejected():
    with pytest.raises(ValidationError, match="schema"):
        FlowDefinition.from_dict(
            {
                "schema": "crewai.flow/v2",
                "name": "FutureSchema",
                "methods": {
                    "begin": {"start": True, "do": {"ref": f"{__name__}:ChainFlow.begin"}}
                },
            }
        )


def test_flow_config_definition_mirrors_flow_fields():
    for name, field in FlowConfigDefinition.model_fields.items():
        assert name in Flow.model_fields
        assert field.get_default(call_default_factory=True) == Flow.model_fields[
            name
        ].get_default(call_default_factory=True)


class DefinitionStoreBackend(FlowPersistence):
    persistence_type: str = "DefinitionStoreBackend"
    store: str = "default"

    saves: ClassVar[dict[str, list[tuple[str, dict[str, Any]]]]] = defaultdict(list)
    pending: ClassVar[dict[str, tuple[dict[str, Any], PendingFeedbackContext]]] = {}

    def init_db(self) -> None:
        pass

    def save_state(self, flow_uuid, method_name, state_data):
        data = state_data if isinstance(state_data, dict) else state_data.model_dump()
        DefinitionStoreBackend.saves[self.store].append((method_name, dict(data)))

    def load_state(self, flow_uuid):
        for _, data in reversed(DefinitionStoreBackend.saves[self.store]):
            if data.get("id") == flow_uuid:
                return data
        return None

    def save_pending_feedback(self, flow_uuid, context, state_data):
        data = state_data if isinstance(state_data, dict) else state_data.model_dump()
        DefinitionStoreBackend.pending[flow_uuid] = (dict(data), context)

    def load_pending_feedback(self, flow_uuid):
        return DefinitionStoreBackend.pending.get(flow_uuid)

    def clear_pending_feedback(self, flow_uuid):
        DefinitionStoreBackend.pending.pop(flow_uuid, None)


def _saved_methods(store):
    return [name for name, _ in DefinitionStoreBackend.saves[store]]


class PersistedFlow(Flow):
    @start()
    def first(self):
        self.state["count"] = self.state.get("count", 0) + 1
        return "one"

    @listen(first)
    def second(self):
        self.state["count"] += 1
        return "two"


def _flow_level_persist_yaml(store):
    return f"""
schema: crewai.flow/v1
name: PersistedFlow
persist:
  enabled: true
  persistence:
    persistence_type: DefinitionStoreBackend
    store: {store}
methods:
  first:
    do:
      ref: {__name__}:PersistedFlow.first
    start: true
  second:
    do:
      ref: {__name__}:PersistedFlow.second
    listen: first
"""


def _method_level_persist_yaml(store):
    return f"""
schema: crewai.flow/v1
name: PersistedFlow
methods:
  first:
    do:
      ref: {__name__}:PersistedFlow.first
    start: true
    persist:
      enabled: true
      persistence:
        persistence_type: DefinitionStoreBackend
        store: {store}
  second:
    do:
      ref: {__name__}:PersistedFlow.second
    listen: first
"""


_CLASS_LEVEL_BACKEND = DefinitionStoreBackend(store="class-decorator")


@persist(_CLASS_LEVEL_BACKEND)
class ClassPersistedFlow(Flow):
    @start()
    def first(self):
        self.state["count"] = self.state.get("count", 0) + 1
        return "one"

    @listen(first)
    def second(self):
        self.state["count"] += 1
        return "two"


_COMBINED_BACKEND = DefinitionStoreBackend(store="combined-decorator")


@persist(_COMBINED_BACKEND)
class CombinedPersistedFlow(Flow):
    @start()
    @persist(_COMBINED_BACKEND)
    def first(self):
        return "one"

    @listen(first)
    def second(self):
        return "two"


class MethodPersistedFlow(Flow):
    @start()
    @persist(DefinitionStoreBackend(store="method-decorator"))
    def first(self):
        self.state["count"] = self.state.get("count", 0) + 1
        return "one"

    @listen(first)
    def second(self):
        self.state["count"] += 1
        return "two"


def test_flow_level_persist_from_yaml_saves_once_per_method():
    yaml_str = _flow_level_persist_yaml("yaml-flow-level")
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    result = flow.kickoff()

    assert result == "two"
    assert _saved_methods("yaml-flow-level") == ["first", "second"]
    _, final_save = DefinitionStoreBackend.saves["yaml-flow-level"][-1]
    assert final_save["count"] == 2
    assert final_save["id"] == flow.state["id"]


def test_method_level_persist_from_yaml_saves_only_that_method():
    yaml_str = _method_level_persist_yaml("yaml-method-level")
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow.kickoff()

    assert _saved_methods("yaml-method-level") == ["first"]
    _, save = DefinitionStoreBackend.saves["yaml-method-level"][0]
    assert save["count"] == 1


def test_method_level_persist_disabled_wins_over_flow_level():
    yaml_str = f"""
schema: crewai.flow/v1
name: PersistedFlow
persist:
  enabled: true
  persistence:
    persistence_type: DefinitionStoreBackend
    store: yaml-opt-out
methods:
  first:
    do:
      ref: {__name__}:PersistedFlow.first
    start: true
  second:
    do:
      ref: {__name__}:PersistedFlow.second
    listen: first
    persist:
      enabled: false
"""
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow.kickoff()

    assert _saved_methods("yaml-opt-out") == ["first"]


def test_persist_restore_by_id_from_yaml():
    yaml_str = _flow_level_persist_yaml("yaml-restore")

    flow1 = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow1.kickoff()
    assert flow1.state["count"] == 2

    flow2 = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow2.kickoff(inputs={"id": flow1.state["id"]})
    assert flow2.state["count"] == 4


def test_combined_class_and_method_persist_saves_once_per_method():
    before = len(DefinitionStoreBackend.saves["combined-decorator"])
    CombinedPersistedFlow().kickoff()

    assert _saved_methods("combined-decorator")[before:] == ["first", "second"]


def test_method_level_persist_decorator_saves_only_that_method():
    before = len(DefinitionStoreBackend.saves["method-decorator"])
    MethodPersistedFlow().kickoff()

    assert _saved_methods("method-decorator")[before:] == ["first"]


def test_round_trip_persist_equivalence():
    definition = FlowDefinition.from_yaml(ClassPersistedFlow.flow_definition().to_yaml())

    before = len(DefinitionStoreBackend.saves["class-decorator"])
    flow = Flow.from_definition(definition)
    flow.kickoff()

    assert _saved_methods("class-decorator")[before:] == ["first", "second"]


def test_method_persist_backend_overrides_flow_level_backend_from_yaml():
    yaml_str = f"""
schema: crewai.flow/v1
name: PersistedFlow
persist:
  enabled: true
  persistence:
    persistence_type: DefinitionStoreBackend
    store: yaml-mixed-flow
methods:
  first:
    do:
      ref: {__name__}:PersistedFlow.first
    start: true
  second:
    do:
      ref: {__name__}:PersistedFlow.second
    listen: first
    persist:
      enabled: true
      persistence:
        persistence_type: DefinitionStoreBackend
        store: yaml-mixed-method
"""
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))
    flow.kickoff()

    assert _saved_methods("yaml-mixed-flow") == ["first"]
    assert _saved_methods("yaml-mixed-method") == ["second"]


def test_method_persist_decorator_overrides_class_level_backend():
    @persist(DefinitionStoreBackend(store="mixed-class"))
    class MixedPersistedFlow(Flow):
        @start()
        @persist(DefinitionStoreBackend(store="mixed-method"))
        def first(self):
            return "one"

        @listen(first)
        def second(self):
            return "two"

    MixedPersistedFlow().kickoff()

    assert _saved_methods("mixed-method") == ["first"]
    assert _saved_methods("mixed-class") == ["second"]


def test_instance_persistence_overrides_definition_backend():
    before = len(DefinitionStoreBackend.saves["method-decorator"])
    flow = MethodPersistedFlow(
        persistence=DefinitionStoreBackend(store="instance-override")
    )
    flow.kickoff()

    assert _saved_methods("instance-override") == ["first"]
    assert len(DefinitionStoreBackend.saves["method-decorator"]) == before


def test_resume_synthetic_completion_persists():
    backend = DefinitionStoreBackend(store="resume-synthetic")

    class ResumableFlow(Flow):
        @start()
        @persist(DefinitionStoreBackend(store="resume-synthetic"))
        @human_feedback(message="Review:")
        def generate(self):
            return "content"

        @listen(generate)
        def process(self, result):
            return "done"

    context = PendingFeedbackContext(
        flow_id="resume-persist-1",
        flow_class="ResumableFlow",
        method_name="generate",
        method_output="content",
        message="Review:",
    )
    backend.save_pending_feedback(
        "resume-persist-1", context, {"id": "resume-persist-1"}
    )

    flow = ResumableFlow.from_pending("resume-persist-1", backend)
    result = flow.resume("looks good")

    assert result == "done"
    assert _saved_methods("resume-synthetic") == ["generate"]


class ReviewFlow(Flow):
    @start()
    @human_feedback(
        message="Review the draft:",
        emit=["approved", "rejected"],
        llm="gpt-4o-mini",
        default_outcome="rejected",
    )
    def draft(self):
        return "draft-content"

    @listen("approved")
    def publish(self):
        return f"published:{self.last_human_feedback.feedback}"

    @listen("rejected")
    def discard(self):
        return "discarded"


REVIEW_YAML = f"""
schema: crewai.flow/v1
name: ReviewFlow
methods:
  draft:
    do:
      ref: {__name__}:ReviewFlow.draft
    start: true
    human_feedback:
      message: "Review the draft:"
      emit: [approved, rejected]
      llm: gpt-4o-mini
      default_outcome: rejected
  publish:
    do:
      ref: {__name__}:ReviewFlow.publish
    listen: approved
  discard:
    do:
      ref: {__name__}:ReviewFlow.discard
    listen: rejected
"""


def _pending_generate(flow):
    return "content"


def _pending_process(flow, result):
    return f"resumed:{result.feedback}"


class PausingProvider:
    def request_feedback(self, context, flow):
        raise HumanFeedbackPending(context=context)


PENDING_REVIEW_YAML = f"""
schema: crewai.flow/v1
name: PendingReviewFlow
persist:
  enabled: true
  persistence:
    persistence_type: DefinitionStoreBackend
    store: hitl-pending
methods:
  generate:
    do:
      ref: {__name__}:_pending_generate
    start: true
    human_feedback:
      message: "Review:"
      provider: {__name__}:PausingProvider
  process:
    do:
      ref: {__name__}:_pending_process
    listen: generate
"""


def test_human_feedback_from_yaml_default_outcome_routes():
    flow = Flow.from_definition(FlowDefinition.from_yaml(REVIEW_YAML))

    with patch.object(flow, "_request_human_feedback", return_value="") as request:
        result = flow.kickoff()

    assert result == "discarded"
    assert request.call_count == 1
    assert flow.last_human_feedback.outcome == "rejected"
    assert flow.last_human_feedback.output == "draft-content"


def test_human_feedback_from_yaml_collapses_and_routes():
    flow = Flow.from_definition(FlowDefinition.from_yaml(REVIEW_YAML))

    with (
        patch.object(flow, "_request_human_feedback", return_value="ship it"),
        patch.object(flow, "_collapse_to_outcome", return_value="approved"),
    ):
        result = flow.kickoff()

    assert result == "published:ship it"
    assert [r.outcome for r in flow.human_feedback_history] == ["approved"]


def test_round_trip_human_feedback_equivalence():
    class_flow = ReviewFlow()
    with patch.object(class_flow, "_request_human_feedback", return_value=""):
        class_result = class_flow.kickoff()

    definition = FlowDefinition.from_yaml(ReviewFlow.flow_definition().to_yaml())
    twin = Flow.from_definition(definition)
    with patch.object(twin, "_request_human_feedback", return_value=""):
        twin_result = twin.kickoff()

    assert twin_result == class_result == "discarded"
    assert (
        twin.last_human_feedback.outcome
        == class_flow.last_human_feedback.outcome
        == "rejected"
    )


def test_human_feedback_pending_and_resume_from_yaml():
    definition = FlowDefinition.from_yaml(PENDING_REVIEW_YAML)

    flow = Flow.from_definition(definition)
    pending = flow.kickoff()

    assert isinstance(pending, HumanFeedbackPending)
    flow_id = pending.context.flow_id
    assert flow_id in DefinitionStoreBackend.pending

    resumed = Flow.from_pending(
        flow_id,
        DefinitionStoreBackend(store="hitl-pending"),
        definition=definition,
    )
    result = resumed.resume("looks good")

    assert result == "resumed:looks good"
    assert resumed.last_human_feedback.feedback == "looks good"
    assert flow_id not in DefinitionStoreBackend.pending


def test_flow_config_provider_fallback_from_yaml():
    yaml_str = f"""
schema: crewai.flow/v1
name: ConfigProviderFlow
methods:
  generate:
    do:
      ref: {__name__}:_pending_generate
    start: true
    human_feedback:
      message: "Review:"
  process:
    do:
      ref: {__name__}:_pending_process
    listen: generate
"""

    class RecordingProvider:
        def __init__(self):
            self.requests = []

        def request_feedback(self, context, flow):
            self.requests.append(context.method_name)
            return "from-config"

    provider = RecordingProvider()
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    previous = flow_config.hitl_provider
    flow_config.hitl_provider = provider
    try:
        result = flow.kickoff()
    finally:
        flow_config.hitl_provider = previous

    assert result == "resumed:from-config"
    assert provider.requests == ["generate"]


# --- PR 7: one resolution story, inert decorator attrs, restore paths ---


def test_runtime_package_reads_no_decorator_attrs():
    import crewai.flow.runtime as flow_runtime

    runtime_dir = Path(flow_runtime.__file__).parent
    forbidden = (
        "__human_feedback_config__",
        "__flow_persistence_config__",
        "__flow_method_definition__",
        "_human_feedback_llm",
    )
    offenders = [
        f"{path.name}: {attr}"
        for path in sorted(runtime_dir.rglob("*.py"))
        for attr in forbidden
        if attr in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_stamped_decorator_attrs_are_inert_at_runtime():
    class StampFreeFlow(Flow):
        @start()
        @persist(DefinitionStoreBackend(store="stamp-free"))
        def first(self):
            return "one"

        @listen(first)
        def second(self, result):
            return f"{result}-two"

    StampFreeFlow.flow_definition()
    stamped = (
        "__flow_method_definition__",
        "__flow_persistence_config__",
        "__human_feedback_config__",
    )
    for name in ("first", "second"):
        wrapper = StampFreeFlow.__dict__[name]
        for attr in stamped:
            if attr in wrapper.__dict__:
                delattr(wrapper, attr)

    result = StampFreeFlow().kickoff()

    assert result == "one-two"
    assert _saved_methods("stamp-free") == ["first"]


def test_class_level_persist_without_instance_kwarg_saves_and_restores():
    before = len(DefinitionStoreBackend.saves["class-decorator"])
    flow = ClassPersistedFlow()
    flow.kickoff()

    assert _saved_methods("class-decorator")[before:] == ["first", "second"]
    assert flow.state["count"] == 2

    resumed = ClassPersistedFlow()
    resumed.kickoff(inputs={"id": flow.state["id"]})
    assert resumed.state["count"] == 4


def test_input_provider_bad_ref_names_field_and_ref():
    with pytest.raises(ValidationError, match="unresolvable input_provider ref"):
        Flow(input_provider="missing_module_xyz:Provider")


class _NeedsArgsProvider:
    def __init__(self, channel):
        self.channel = channel

    def request_feedback(self, context, flow):
        return "ok"


def test_provider_ref_requiring_ctor_args_fails_loudly():
    yaml_str = f"""
schema: crewai.flow/v1
name: BadProviderFlow
methods:
  generate:
    do:
      ref: {__name__}:_pending_generate
    start: true
    human_feedback:
      message: "Review:"
      provider: {__name__}:_NeedsArgsProvider
"""
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    with pytest.raises(
        ValueError, match="cannot instantiate human_feedback.provider ref"
    ):
        flow.kickoff()


def test_unresolvable_provider_ref_names_field_and_ref():
    yaml_str = f"""
schema: crewai.flow/v1
name: BadProviderFlow
methods:
  generate:
    do:
      ref: {__name__}:_pending_generate
    start: true
    human_feedback:
      message: "Review:"
      provider: missing_module_xyz:Provider
"""
    flow = Flow.from_definition(FlowDefinition.from_yaml(yaml_str))

    with pytest.raises(
        ValueError, match="unresolvable human_feedback.provider ref"
    ):
        flow.kickoff()


def _checkpoint_chain_flow(tmp_path):
    from crewai.state.provider.json_provider import JsonProvider
    from crewai.state.runtime import RuntimeState

    definition = FlowDefinition.from_yaml(CHAIN_YAML)
    flow = Flow.from_definition(definition)
    result = flow.kickoff()
    assert result == "confirmed:True"

    state = RuntimeState(root=[flow])
    state._provider = JsonProvider()
    location = state.checkpoint(str(tmp_path))
    return definition, flow, CheckpointConfig(restore_from=location)


def test_from_checkpoint_with_definition_restores_yaml_flow(tmp_path):
    definition, flow, config = _checkpoint_chain_flow(tmp_path)

    restored = Flow.from_checkpoint(config, definition=definition)

    assert restored.state["confirmed"] is True
    assert restored.state["id"] == flow.state["id"]
    assert restored.kickoff() == "confirmed:True"


def test_fork_with_definition_branches_yaml_flow(tmp_path):
    definition, flow, config = _checkpoint_chain_flow(tmp_path)

    forked = Flow.fork(config, branch="alt", definition=definition)

    assert forked.state["id"] != flow.state["id"]
    assert forked.kickoff() == "confirmed:True"


def test_non_dict_state_default_rejected_by_contract():
    yaml_str = """
schema: crewai.flow/v1
name: BadStateFlow
state:
  type: dict
  default: 42
methods: {}
"""
    with pytest.raises(ValidationError, match="default"):
        FlowDefinition.from_yaml(yaml_str)


def test_definition_method_missing_from_class_fails_loudly():
    class VanishingFlow(Flow):
        @start()
        def begin(self):
            return "one"

    VanishingFlow.flow_definition()
    del VanishingFlow.begin

    with pytest.raises(ValueError, match="does not provide: begin"):
        VanishingFlow()
