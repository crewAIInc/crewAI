"""Tests for the agent-hooks control engine (:mod:`crewai.hooks.agent_hooks_engine`).

Helper-level tests run without agent-hooks installed. Governance tests activate
the engine and exercise it through crewAI's real interception seams; they are
skipped when the optional ``agent_hooks`` dependency (and its native core) is
unavailable.
"""

from __future__ import annotations

import asyncio
import json
import threading
import types
from typing import Any

from crewai.hooks.agent_hooks_engine import (
    HAS_AGENT_HOOKS,
    _EmitterLoop,
    _agent_ids,
    _blocked_result,
    _json_safe,
    _model_id,
    _registered_tools,
    _session_id,
    _to_text,
    _transformed_payload,
    disable_agent_hooks,
    use_agent_hooks,
)
from crewai.hooks.contexts import (
    ExecutionStartContext,
    InputContext,
)
from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
    clear_all,
    clear_governor,
    dispatch,
    get_governor,
)
from crewai.hooks.llm_hooks import (
    LLMCallHookContext,
    after_llm_call_reducer,
    before_llm_call_reducer,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    run_after_tool_call_hooks,
    run_before_tool_call_hooks,
)
import pytest


requires_ah = pytest.mark.skipif(
    not HAS_AGENT_HOOKS, reason="agent-hooks (agent_hooks) is not installed"
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Every test starts and ends with an empty registry and no active engine."""
    clear_all()
    yield
    disable_agent_hooks()
    clear_governor()
    clear_all()


# --- lightweight crewAI-shaped fakes ----------------------------------------


def _crew(cid: str = "crew-1", tools: list[Any] | None = None) -> Any:
    return types.SimpleNamespace(id=cid, tools=list(tools or []))


def _agent(aid: str = "agent-1", role: str = "tester") -> Any:
    return types.SimpleNamespace(id=aid, role=role)


def _flow(fid: str = "flow-1") -> Any:
    return types.SimpleNamespace(id=fid)


def _llm(model: str = "gpt-test") -> Any:
    return types.SimpleNamespace(model=model)


def _tool(name: str) -> Any:
    return types.SimpleNamespace(name=name)


def _tool_ctx(
    name: str = "web_search",
    tool_input: dict[str, Any] | None = None,
    tool_result: str | None = None,
    agent: Any = None,
    crew: Any = None,
) -> ToolCallHookContext:
    return ToolCallHookContext(
        tool_name=name,
        tool_input=tool_input if tool_input is not None else {"q": "x"},
        tool=_tool(name),
        agent=agent if agent is not None else _agent(),
        crew=crew if crew is not None else _crew(),
        tool_result=tool_result,
    )


def _llm_ctx(
    messages: list[dict[str, Any]] | None = None, response: str | None = None
) -> LLMCallHookContext:
    return LLMCallHookContext(
        messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
        llm=_llm(),
        agent=_agent(),
        crew=_crew(),
        response=response,
    )


# --- interceptors (agent-hooks) ---------------------------------------------


class _Allow:
    def intercept(self, ctx: Any) -> Any:
        from agent_hooks import Decision, Verdict

        return Verdict(decision=Decision.ALLOW)


class _DenyAt:
    def __init__(self, point: str, reason: str = "denied") -> None:
        self.point = point
        self.reason = reason

    def intercept(self, ctx: Any) -> Any:
        from agent_hooks import Decision, Verdict

        if ctx["interception_point"] == self.point:
            return Verdict.deny(reason=self.reason)
        return Verdict(decision=Decision.ALLOW)


class _TransformAt:
    def __init__(self, point: str, value: Any, path: str = "$target") -> None:
        self.point = point
        self.value = value
        self.path = path

    def intercept(self, ctx: Any) -> Any:
        from agent_hooks import Decision, Transform, Verdict

        if ctx["interception_point"] == self.point:
            return Verdict(
                decision=Decision.TRANSFORM,
                transform=Transform(path=self.path, value=self.value),
            )
        return Verdict(decision=Decision.ALLOW)


def _plain(ctx: Any) -> dict[str, Any]:
    """Render an agent-hooks ``AgentContext`` as a plain nested ``dict``."""
    return json.loads(json.dumps(dict(ctx), default=str))


class _Capture:
    """Records the exact ``AgentContext`` each interceptor call receives, then allows."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def intercept(self, ctx: Any) -> Any:
        from agent_hooks import Decision, Verdict

        self.seen.append(_plain(ctx))
        return Verdict(decision=Decision.ALLOW)

    def at(self, point: str) -> list[dict[str, Any]]:
        return [c for c in self.seen if c.get("interception_point") == point]


# --- helper units (no agent-hooks needed) -----------------------------------


class TestHelpers:
    def test_json_safe_passthrough(self):
        value = {"a": [1, 2.0, "x", True, None]}
        assert _json_safe(value) == value

    def test_json_safe_non_finite_floats(self):
        assert _json_safe(float("inf")) == "inf"
        assert _json_safe(float("nan")) == "nan"

    def test_json_safe_bytes(self):
        assert _json_safe(b"hi") == "hi"

    def test_json_safe_set_is_listified(self):
        assert sorted(_json_safe({3, 1, 2})) == [1, 2, 3]

    def test_json_safe_model_dump(self):
        class Model:
            def model_dump(self, mode: str = "python") -> dict[str, int]:
                return {"k": 1}

        assert _json_safe(Model()) == {"k": 1}

    def test_json_safe_unknown_becomes_str(self):
        class Opaque:
            pass

        assert isinstance(_json_safe(Opaque()), str)

    def test_json_safe_nested_object(self):
        class Model:
            def model_dump(self, mode: str = "python") -> dict[str, Any]:
                return {"n": 1}

        assert _json_safe({"outer": Model()}) == {"outer": {"n": 1}}

    def test_agent_ids_none(self):
        assert _agent_ids(None) == ("crewai-agent", None)

    def test_agent_ids_from_id_and_role(self):
        assert _agent_ids(_agent("a", "r")) == ("a", "r")

    def test_agent_ids_role_fallback(self):
        agent = types.SimpleNamespace(role="analyst")
        assert _agent_ids(agent) == ("analyst", "analyst")

    def test_session_id_prefers_crew(self):
        assert _session_id(crew=_crew("c9")) == "c9"

    def test_session_id_agent_fallback(self):
        assert _session_id(agent=_agent("a2")) == "agent:a2"

    def test_session_id_default(self):
        assert _session_id() == "crewai-session"

    def test_model_id(self):
        assert _model_id(_llm("m")) == "m"

    def test_model_id_unknown(self):
        assert _model_id(object()) == "unknown"

    def test_blocked_result(self):
        assert _blocked_result("bad") == "[blocked by agent-hooks: bad]"

    def test_to_text_passthrough_and_json(self):
        assert _to_text("hi") == "hi"
        assert _to_text({"a": 1}) == '{"a": 1}'

    def test_transformed_payload_variants(self):
        assert _transformed_payload({"content": 5, "role": "user"}) == 5
        assert _transformed_payload({"x": 1}) == {"x": 1}
        assert _transformed_payload("z") == "z"

    def test_registered_tools(self):
        crew = _crew(tools=[_tool("a"), _tool("b")])
        assert _registered_tools(crew) == ["a", "b"]


# --- emitter loop internals -------------------------------------------------


class TestEmitterLoop:
    def test_close_releases_pending_emission(self):
        """``close()`` unblocks a thread waiting on a never-resolving emission."""
        loop = _EmitterLoop()
        started = threading.Event()

        async def hang() -> None:
            started.set()
            await asyncio.Event().wait()  # never resolves

        def worker() -> None:
            try:
                loop.run(hang(), None)  # emit_timeout=None -> blocks until close
            except BaseException:
                pass

        t = threading.Thread(target=worker, name="pending-emit", daemon=True)
        t.start()
        assert started.wait(timeout=5.0), "emission never started"

        loop.close()  # must release the blocked emission
        t.join(timeout=2.0)
        assert not t.is_alive()


# --- tool-call governance (through the real seam) ---------------------------


@requires_ah
class TestToolGovernance:
    def test_allow_proceeds_unchanged(self):
        engine = use_agent_hooks(_Allow())
        try:
            ctx = _tool_ctx(tool_input={"q": "x"})
            assert run_before_tool_call_hooks(ctx) is False
            assert ctx.tool_input == {"q": "x"}
        finally:
            engine.close()

    def test_deny_blocks_call(self):
        engine = use_agent_hooks(_DenyAt("pre_tool_call", "nope"))
        try:
            assert run_before_tool_call_hooks(_tool_ctx()) is True
        finally:
            engine.close()

    def test_transform_rewrites_args_in_place(self):
        engine = use_agent_hooks(_TransformAt("pre_tool_call", {"url": "https://safe"}))
        try:
            ctx = _tool_ctx(tool_input={"url": "http://evil"})
            original = ctx.tool_input
            run_before_tool_call_hooks(ctx)
            assert ctx.tool_input == {"url": "https://safe"}
            assert ctx.tool_input is original  # crewAI requires in-place mutation
        finally:
            engine.close()

    def test_post_transform_rewrites_result(self):
        engine = use_agent_hooks(_TransformAt("post_tool_call", "REDACTED"))
        try:
            ctx = _tool_ctx(tool_result="secret data")
            assert run_after_tool_call_hooks(ctx) == "REDACTED"
        finally:
            engine.close()

    def test_post_deny_fails_closed(self):
        engine = use_agent_hooks(_DenyAt("post_tool_call", "bad"))
        try:
            ctx = _tool_ctx(tool_result="secret data")
            assert run_after_tool_call_hooks(ctx) == "[blocked by agent-hooks: bad]"
        finally:
            engine.close()

    def test_native_hook_runs_before_engine(self):
        """A native before_tool_call hook mutates input; the engine sees it."""
        from crewai.hooks.dispatch import register

        def widen(ctx: Any) -> None:
            ctx.tool_input["added"] = True

        register(InterceptionPoint.PRE_TOOL_CALL, widen)
        engine = use_agent_hooks(_Allow())
        try:
            ctx = _tool_ctx(tool_input={"q": "x"})
            run_before_tool_call_hooks(ctx)
            assert ctx.tool_input == {"q": "x", "added": True}
        finally:
            engine.close()

    def test_cyclic_tool_input_fails_closed(self):
        """A cyclic ``tool_input`` is normalized, not passed through ungoverned.

        A self-referential arg would raise ``RecursionError`` while the context
        is built; the engine must still let the policy block the call (fail
        closed) instead of proceeding ungoverned.
        """
        engine = use_agent_hooks(_DenyAt("pre_tool_call", "policy: tool blocked"))
        try:
            assert run_before_tool_call_hooks(_tool_ctx()) is True  # control deny
            cyclic: dict[str, Any] = {}
            cyclic["self"] = cyclic
            assert run_before_tool_call_hooks(_tool_ctx(tool_input=cyclic)) is True
        finally:
            engine.close()

    def test_pre_and_post_tool_call_ids_correlate(self):
        """The separate pre/post contexts for one invocation share a call id."""
        cap = _Capture()
        engine = use_agent_hooks(cap)
        crew = _crew("crew-1")
        agent = _agent("agent-A")
        try:
            run_before_tool_call_hooks(
                _tool_ctx(tool_input={"q": "x"}, agent=agent, crew=crew)
            )
            run_after_tool_call_hooks(
                _tool_ctx(
                    tool_input={"q": "x"},
                    tool_result="done",
                    agent=agent,
                    crew=crew,
                )
            )
            pre_id = cap.at("pre_tool_call")[0]["tool_call"]["id"]
            post_id = cap.at("post_tool_call")[0]["tool_call"]["id"]
            assert pre_id == post_id
        finally:
            engine.close()


# --- model-call governance (direct-call seam via dispatch) ------------------


@requires_ah
class TestModelGovernance:
    def test_pre_deny_raises(self):
        engine = use_agent_hooks(_DenyAt("pre_model_call"))
        try:
            with pytest.raises(HookAborted):
                dispatch(
                    InterceptionPoint.PRE_MODEL_CALL,
                    _llm_ctx(),
                    reducer=before_llm_call_reducer,
                )
        finally:
            engine.close()

    def test_pre_transform_rewrites_messages(self):
        replacement = [{"role": "user", "content": "safe"}]
        engine = use_agent_hooks(_TransformAt("pre_model_call", replacement))
        try:
            ctx = _llm_ctx(messages=[{"role": "user", "content": "danger"}])
            original = ctx.messages
            dispatch(
                InterceptionPoint.PRE_MODEL_CALL, ctx, reducer=before_llm_call_reducer
            )
            assert ctx.messages == replacement
            assert ctx.messages is original
        finally:
            engine.close()

    def test_post_transform_rewrites_response(self):
        new_response = {"content": "clean", "tool_calls": [], "finish_reason": "stop"}
        engine = use_agent_hooks(_TransformAt("post_model_call", new_response))
        try:
            ctx = _llm_ctx(response="raw")
            dispatch(
                InterceptionPoint.POST_MODEL_CALL, ctx, reducer=after_llm_call_reducer
            )
            assert ctx.response == "clean"
        finally:
            engine.close()

    def test_post_model_call_exposes_requested_tool_calls(self):
        """An interceptor at ``post_model_call`` sees the model's tool calls."""
        cap = _Capture()
        engine = use_agent_hooks(cap)
        try:
            requested = [{"id": "tc-1", "name": "shell", "args": {"cmd": "rm -rf /"}}]
            ctx = types.SimpleNamespace(
                messages=[{"role": "user", "content": "hi"}],
                llm=_llm(),
                agent=_agent(),
                crew=_crew(),
                response={"content": "", "tool_calls": requested},
            )
            dispatch(InterceptionPoint.POST_MODEL_CALL, ctx)
            seen = cap.at("post_model_call")
            assert seen, "post_model_call was not emitted"
            assert seen[0]["response"]["tool_calls"] == requested
        finally:
            engine.close()


# --- execution-boundary governance ------------------------------------------


@requires_ah
class TestBoundaryGovernance:
    def test_input_deny_raises(self):
        engine = use_agent_hooks(_DenyAt("input"))
        try:
            ctx = InputContext(crew=_crew(), inputs={"t": "x"}, payload={"t": "x"})
            with pytest.raises(HookAborted):
                dispatch(InterceptionPoint.INPUT, ctx)
        finally:
            engine.close()

    def test_input_transform_replaces_payload(self):
        engine = use_agent_hooks(
            _TransformAt("input", {"t": "safe"}, path="$target.content")
        )
        try:
            ctx = InputContext(crew=_crew(), inputs={"t": "x"}, payload={"t": "x"})
            dispatch(InterceptionPoint.INPUT, ctx)
            assert ctx.payload == {"t": "safe"}
        finally:
            engine.close()

    def test_execution_start_deny_raises(self):
        engine = use_agent_hooks(_DenyAt("agent_startup"))
        try:
            ctx = ExecutionStartContext(crew=_crew(), inputs={}, payload={})
            with pytest.raises(HookAborted):
                dispatch(InterceptionPoint.EXECUTION_START, ctx)
        finally:
            engine.close()


# --- session & identity scoping ---------------------------------------------


@requires_ah
class TestSessionScoping:
    def test_builder_preserves_per_agent_identity(self):
        """Different agents in one crew keep their own id in the record.

        The first (agent-less) ``EXECUTION_START`` creates the cached session
        builder; every later emission must still be attributed to the agent
        that acted, not the session's first agent.
        """
        cap = _Capture()
        engine = use_agent_hooks(cap)
        crew = _crew("crew-1")
        try:
            dispatch(
                InterceptionPoint.EXECUTION_START,
                ExecutionStartContext(crew=crew, inputs={}, payload={}),
            )
            run_before_tool_call_hooks(_tool_ctx(agent=_agent("agent-A"), crew=crew))
            run_before_tool_call_hooks(_tool_ctx(agent=_agent("agent-B"), crew=crew))
            ids = [c["agent"]["id"] for c in cap.at("pre_tool_call")]
            assert ids == ["agent-A", "agent-B"]
        finally:
            engine.close()

    def test_distinct_flows_get_distinct_sessions(self):
        """Each flow run maps to its own agent-hooks session."""
        cap = _Capture()
        engine = use_agent_hooks(cap)
        try:
            dispatch(
                InterceptionPoint.INPUT,
                InputContext(
                    flow=_flow("flow-1"), inputs={"t": "1"}, payload={"t": "1"}
                ),
            )
            dispatch(
                InterceptionPoint.INPUT,
                InputContext(
                    flow=_flow("flow-2"), inputs={"t": "2"}, payload={"t": "2"}
                ),
            )
            sessions = [c["session"]["id"] for c in cap.at("input")]
            assert sessions == ["flow-1", "flow-2"]
        finally:
            engine.close()


# --- engine lifecycle & configuration ---------------------------------------


@requires_ah
class TestEngineLifecycle:
    def test_use_agent_hooks_installs_governor(self):
        engine = use_agent_hooks(_Allow())
        try:
            assert get_governor() is not None
        finally:
            engine.close()

    def test_records_populated_and_drainable(self):
        engine = use_agent_hooks(_Allow())
        try:
            run_before_tool_call_hooks(_tool_ctx())
            assert len(engine.records) == 1
            assert len(engine.take_records()) == 1
            assert engine.records == []
        finally:
            engine.close()

    def test_record_sink_receives_every_decision(self):
        seen: list[Any] = []
        engine = use_agent_hooks(_Allow(), record_sink=seen.append)
        try:
            run_before_tool_call_hooks(_tool_ctx())
            assert len(seen) == 1
        finally:
            engine.close()

    def test_activate_is_idempotent(self):
        engine = use_agent_hooks(_Allow())
        try:
            engine.activate()
            assert get_governor() is not None
        finally:
            engine.close()

    def test_close_clears_governor(self):
        engine = use_agent_hooks(_Allow())
        assert get_governor() is not None
        engine.close()
        assert get_governor() is None

    def test_points_subset_limits_governance(self):
        engine = use_agent_hooks(_Allow(), points=[InterceptionPoint.PRE_TOOL_CALL])
        try:
            assert engine.adapter_for(InterceptionPoint.PRE_TOOL_CALL) is not None
            assert engine.adapter_for(InterceptionPoint.POST_TOOL_CALL) is None
        finally:
            engine.close()

    def test_context_manager_activates_and_closes(self):
        with use_agent_hooks(_Allow()):
            assert get_governor() is not None
        assert get_governor() is None

    def test_second_use_supersedes_first(self):
        first = use_agent_hooks(_Allow())
        second = use_agent_hooks(_Allow())
        try:
            assert get_governor() == second.adapter_for
        finally:
            second.close()
            first.close()

    def test_evaluate_only_records_but_never_blocks(self):
        from agent_hooks import EnforcementMode

        engine = use_agent_hooks(
            _DenyAt("pre_tool_call"), mode=EnforcementMode.EVALUATE_ONLY
        )
        try:
            assert run_before_tool_call_hooks(_tool_ctx()) is False
            assert len(engine.records) == 1
        finally:
            engine.close()

    def test_zero_interceptors_fails_closed(self):
        engine = use_agent_hooks()
        try:
            assert run_before_tool_call_hooks(_tool_ctx()) is True
        finally:
            engine.close()


@pytest.mark.skipif(HAS_AGENT_HOOKS, reason="agent-hooks IS installed")
def test_use_agent_hooks_raises_actionable_error_without_agent_hooks():
    with pytest.raises(ImportError, match="agent-hooks"):
        use_agent_hooks()
