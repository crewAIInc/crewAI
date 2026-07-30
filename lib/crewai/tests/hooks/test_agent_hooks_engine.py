"""Tests for the agent-hooks control engine (:mod:`crewai.hooks.agent_hooks_engine`).

Helper-level tests run without agent-hooks installed. Governance tests activate
the engine and exercise it through crewAI's real interception seams; they are
skipped when the optional ``agent_hooks`` dependency (and its native core) is
unavailable.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
import gc
from importlib.resources import files
import json
import threading
import types
from typing import Any

from crewai.hooks.agent_hooks_engine import (
    DEFAULT_MAX_RECORDS,
    HAS_AGENT_HOOKS,
    _EmitterLoop,
    _agent_ids,
    _blocked_result,
    _correlation_ids,
    _json_safe,
    _log_identifier,
    _model_id,
    _registered_tools,
    _session_id,
    _to_text,
    _transformed_payload,
    active_engine,
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
    register,
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
import jsonschema
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
    call_id: str | None = None,
    was_blocked: bool = False,
) -> ToolCallHookContext:
    return ToolCallHookContext(
        tool_name=name,
        tool_input=tool_input if tool_input is not None else {"q": "x"},
        tool=_tool(name),
        agent=agent if agent is not None else _agent(),
        crew=crew if crew is not None else _crew(),
        tool_result=tool_result,
        call_id=call_id,
        was_blocked=was_blocked,
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
            return Verdict(decision=Decision.DENY, reason=self.reason)
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


class _DenyToolNameAtPostModel:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def intercept(self, ctx: Any) -> Any:
        from agent_hooks import Decision, Verdict

        if ctx["interception_point"] == "post_model_call":
            tool_calls = ctx["response"]["tool_calls"]
            if any(call.get("name") == self.tool_name for call in tool_calls):
                return Verdict(decision=Decision.DENY, reason="tool denied")
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

    def test_correlation_ids_are_payload_free_and_log_safe(self) -> None:
        ctx = {
            "session": {"id": "session\nforged"},
            "request_id": "request-1",
            "tool_call": {"id": "call-1", "args": {"secret": "ignored"}},
        }

        assert _correlation_ids(ctx) == (
            "session\\nforged",
            "request-1",
            "call-1",
        )
        assert _log_identifier("x" * 200) == "x" * 128
        assert _log_identifier({"secret": "ignored"}) is None

    def test_registered_tools(self):
        crew = _crew(tools=[_tool("a"), _tool("b")])
        assert _registered_tools(crew) == ["a", "b"]

    def test_registered_tools_include_all_crew_agents(self):
        crew = types.SimpleNamespace(
            agents=[
                types.SimpleNamespace(tools=[_tool("search"), _tool("shared")]),
                types.SimpleNamespace(tools=[_tool("shell"), _tool("shared")]),
            ]
        )
        assert _registered_tools(crew) == ["search", "shared", "shell"]


# --- emitter loop internals -------------------------------------------------


class TestEmitterLoop:
    def test_admission_waiter_budget_is_bounded(self) -> None:
        """Excess submissions fail closed instead of joining an unbounded queue."""
        loop = _EmitterLoop(max_waiters=1)

        async def complete() -> str:
            return "done"

        assert loop._waiter_slots.acquire(blocking=False)
        try:
            with pytest.raises(RuntimeError, match="admission queue is full"):
                loop.run(complete(), None)
        finally:
            loop._waiter_slots.release()
            loop.close()

    def test_close_wakes_submission_waiting_for_admission(self) -> None:
        """A caller cannot remain blocked on admission after shutdown."""

        class BlockingAdmission:
            def __init__(self) -> None:
                self.waiting = threading.Event()
                self.allowed = threading.Event()

            def acquire(
                self,
                blocking: bool = True,
                timeout: float | None = None,
            ) -> bool:
                self.waiting.set()
                if not blocking:
                    return self.allowed.is_set()
                return self.allowed.wait(timeout)

            def release(self) -> None:
                self.allowed.set()

        loop = _EmitterLoop()
        admission = BlockingAdmission()
        loop._admission = admission
        runner_done = threading.Event()

        async def complete() -> str:
            return "done"

        def run() -> None:
            try:
                loop.run(complete(), None)
            except BaseException:
                pass
            finally:
                runner_done.set()

        runner = threading.Thread(target=run)
        try:
            runner.start()
            assert admission.waiting.wait(timeout=2.0)
            loop.close()
            assert runner_done.wait(timeout=2.0)
        finally:
            admission.allowed.set()
            runner.join(timeout=2.0)
            if loop._thread.is_alive():
                loop.close()

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

    def test_timeout_cancels_and_releases_admission(self) -> None:
        """A timed-out cancellable emission exits before another is admitted."""
        loop = _EmitterLoop()
        started = threading.Event()
        exited = threading.Event()

        async def hang() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                exited.set()

        async def complete() -> str:
            return "done"

        try:
            with pytest.raises(FutureTimeoutError):
                loop.run(hang(), 0.1)
            assert started.is_set()
            assert exited.wait(timeout=2.0), "timed-out emission did not unwind"
            assert loop.run(complete(), 1.0) == "done"
        finally:
            loop.close()

    def test_cancellation_resistant_timeout_keeps_admission_bounded(self) -> None:
        """Timed-out work retains the sole slot until it actually exits."""
        loop = _EmitterLoop()
        started = threading.Event()
        cancellation_seen = threading.Event()
        exited = threading.Event()
        release_future: list[asyncio.Future[None]] = []

        async def resist_cancellation() -> None:
            release = asyncio.get_running_loop().create_future()
            release_future.append(release)
            started.set()
            try:
                await asyncio.shield(release)
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release
            finally:
                exited.set()

        async def complete() -> str:
            return "done"

        try:
            with pytest.raises(FutureTimeoutError):
                loop.run(resist_cancellation(), 0.1)
            assert started.is_set()
            assert cancellation_seen.wait(timeout=2.0)

            with pytest.raises(FutureTimeoutError, match="admission"):
                loop.run(complete(), 0.1)

            loop._loop.call_soon_threadsafe(release_future[0].set_result, None)
            assert exited.wait(timeout=2.0), "resistant emission never exited"
            assert loop.run(complete(), 1.0) == "done"
        finally:
            loop.close()

    def test_close_cannot_race_past_submission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Close waits for atomic submission and leaves no blocked caller."""
        original_submit = asyncio.run_coroutine_threadsafe
        submitting = threading.Event()
        release_submission = threading.Event()
        first_submission = True

        def submit(coro: Any, loop: asyncio.AbstractEventLoop) -> Any:
            nonlocal first_submission
            if first_submission:
                first_submission = False
                submitting.set()
                assert release_submission.wait(timeout=2.0)
            return original_submit(coro, loop)

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
        loop = _EmitterLoop()
        runner_done = threading.Event()
        close_done = threading.Event()
        close_errors: list[BaseException] = []

        async def complete() -> str:
            return "done"

        def run() -> None:
            try:
                loop.run(complete(), None)
            except BaseException:
                pass
            finally:
                runner_done.set()

        def close() -> None:
            try:
                loop.close(timeout=2.0)
            except BaseException as error:
                close_errors.append(error)
            finally:
                close_done.set()

        runner = threading.Thread(target=run)
        closer = threading.Thread(target=close)
        runner.start()
        assert submitting.wait(timeout=2.0)
        closer.start()
        release_submission.set()

        assert runner_done.wait(timeout=2.0)
        assert close_done.wait(timeout=2.0)
        runner.join()
        closer.join()
        assert close_errors == []

    def test_failed_close_is_observable_and_retryable(self) -> None:
        """Cancellation-resistant work prevents false successful shutdown."""
        loop = _EmitterLoop()
        started = threading.Event()
        cancelled = threading.Event()
        exited = threading.Event()
        release_event: list[asyncio.Event] = []

        async def resist_cancellation() -> None:
            release = asyncio.Event()
            release_event.append(release)
            started.set()
            try:
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled.set()
            finally:
                exited.set()

        def run() -> None:
            try:
                loop.run(resist_cancellation(), None)
            except BaseException:
                pass

        runner = threading.Thread(target=run)
        runner.start()
        assert started.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="did not drain"):
            loop.close(timeout=0.1)
        assert cancelled.is_set()

        loop._loop.call_soon_threadsafe(release_event[0].set)
        assert exited.wait(timeout=2.0)
        loop.close(timeout=2.0)
        runner.join(timeout=2.0)
        assert not runner.is_alive()


# --- tool-call governance (through the real seam) ---------------------------


@requires_ah
class TestToolGovernance:
    def test_direct_tool_post_hook_is_attempted_once_on_abort(self) -> None:
        """A post-hook abort cannot trigger a duplicate post dispatch."""
        from crewai.hooks import register_after_tool_call_hook
        from crewai.llm import LLM

        calls = 0

        def abort_post(context: ToolCallHookContext) -> None:
            nonlocal calls
            calls += 1
            raise HookAborted(reason="stop")

        tool_call = types.SimpleNamespace(
            id="call-1",
            function=types.SimpleNamespace(name="tool", arguments="{}"),
        )
        register_after_tool_call_hook(abort_post)

        with pytest.raises(HookAborted):
            LLM._handle_tool_call(
                types.SimpleNamespace(),
                [tool_call],
                {"tool": lambda: "result"},
            )

        assert calls == 1

    def test_invalid_direct_llm_tool_args_do_not_emit_orphan_post(self) -> None:
        """A parse failure before the pre hook cannot produce a post record."""
        from crewai.llm import LLM

        tool_call = types.SimpleNamespace(
            id="call-1",
            function=types.SimpleNamespace(
                name="dangerous_tool",
                arguments="{invalid-json",
            ),
        )
        cap = _Capture()
        engine = use_agent_hooks(cap)
        try:
            result = LLM._handle_tool_call(
                types.SimpleNamespace(),
                [tool_call],
                {"dangerous_tool": lambda: "executed"},
            )

            assert result is None
            assert cap.at("pre_tool_call") == []
            assert cap.at("post_tool_call") == []
        finally:
            engine.close()

    def test_litellm_tool_execution_is_governed(self):
        """A LiteLLM tool request cannot bypass a pre-tool deny."""
        from crewai.llm import LLM

        invoked = False

        def dangerous_tool() -> str:
            nonlocal invoked
            invoked = True
            return "executed"

        tool_call = types.SimpleNamespace(
            id="call-1",
            function=types.SimpleNamespace(name="dangerous_tool", arguments="{}"),
        )
        engine = use_agent_hooks(_DenyAt("pre_tool_call", "blocked"))
        try:
            llm = LLM(model="gpt-4o-mini", is_litellm=True)
            llm._handle_tool_call(
                [tool_call], {"dangerous_tool": dangerous_tool}
            )
            assert invoked is False
        finally:
            engine.close()

    def test_base_llm_native_tool_execution_is_governed(self):
        """A provider-native tool cannot bypass a pre-tool deny."""
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(
                self,
                messages: Any,
                tools: Any = None,
                callbacks: Any = None,
                available_functions: Any = None,
                from_task: Any = None,
                from_agent: Any = None,
                response_model: Any = None,
            ) -> str:
                return "unused"

            def execute_native_tool(
                self,
                function_name: str,
                function_args: dict[str, Any],
                available_functions: dict[str, Any],
            ) -> str | None:
                return self._handle_tool_execution(
                    function_name=function_name,
                    function_args=function_args,
                    available_functions=available_functions,
                )

        invoked = False

        def dangerous_tool() -> str:
            nonlocal invoked
            invoked = True
            return "executed"

        engine = use_agent_hooks(_DenyAt("pre_tool_call", "blocked"))
        try:
            llm = TestLLM(model="test")
            llm.execute_native_tool(
                function_name="dangerous_tool",
                function_args={},
                available_functions={"dangerous_tool": dangerous_tool},
            )
            assert invoked is False
        finally:
            engine.close()

    def test_base_llm_post_hook_is_attempted_once_on_abort(self) -> None:
        """A provider-native post-hook abort cannot trigger a second dispatch."""
        from crewai.hooks import register_after_tool_call_hook
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(
                self,
                messages: Any,
                tools: Any = None,
                callbacks: Any = None,
                available_functions: Any = None,
                from_task: Any = None,
                from_agent: Any = None,
                response_model: Any = None,
            ) -> str:
                return "unused"

        calls = 0

        def abort_post(context: ToolCallHookContext) -> None:
            nonlocal calls
            calls += 1
            raise HookAborted(reason="stop")

        register_after_tool_call_hook(abort_post)
        llm = TestLLM(model="test")

        with pytest.raises(HookAborted):
            llm._handle_tool_execution("tool", {}, {"tool": lambda: "result"})

        assert calls == 1

    def test_base_llm_native_tool_failure_emits_error_context(self):
        """A failed native tool still emits a correlated error result."""
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(
                self,
                messages: Any,
                tools: Any = None,
                callbacks: Any = None,
                available_functions: Any = None,
                from_task: Any = None,
                from_agent: Any = None,
                response_model: Any = None,
            ) -> str:
                return "unused"

            def execute_native_tool(self, function: Any) -> str | None:
                return self._handle_tool_execution(
                    "dangerous_tool", {}, {"dangerous_tool": function}
                )

        def failing_tool() -> str:
            raise RuntimeError("failed")

        cap = _Capture()
        engine = use_agent_hooks(cap)
        try:
            assert TestLLM(model="test").execute_native_tool(failing_tool) is None
            post = cap.at("post_tool_call")[0]
            assert post["tool_result"]["is_error"] is True
            assert post["tool_call"]["id"]
        finally:
            engine.close()

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

    def test_invalid_tool_transform_shape_blocks_call(self):
        engine = use_agent_hooks(_TransformAt("pre_tool_call", "invalid"))
        try:
            assert run_before_tool_call_hooks(_tool_ctx()) is True
        finally:
            engine.close()

    def test_approved_escalation_preserves_prior_transform(self):
        """An approval must not discard a transform already enforced by the SDK."""
        from agent_hooks import (
            ApprovalOutcome,
            ApprovalResolution,
            CompositionConfig,
            Decision,
            OnApproval,
            Verdict,
        )

        class Escalate:
            def intercept(self, ctx: Any) -> Any:
                return Verdict.escalate(reason="review transformed call")

        class Approver:
            def resolve(self, request: Any) -> Any:
                return ApprovalResolution(
                    outcome=ApprovalOutcome.APPROVE,
                    context_identity=request.context_identity,
                    verdict=Verdict(decision=Decision.ALLOW),
                )

        engine = use_agent_hooks(
            _TransformAt("pre_tool_call", {"url": "https://safe"}),
            Escalate(),
            resolver=Approver(),
            composition=CompositionConfig.first_deny(OnApproval.STOP),
        )
        try:
            ctx = _tool_ctx(tool_input={"url": "http://evil"})
            assert run_before_tool_call_hooks(ctx) is False
            assert ctx.tool_input == {"url": "https://safe"}
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

    def test_pre_block_does_not_emit_post_tool_record(self):
        """A blocked pre-tool action has no corresponding agent-hooks post."""
        engine = use_agent_hooks(_DenyAt("pre_tool_call", "blocked"))
        try:
            call_id = "call-1"
            assert run_before_tool_call_hooks(_tool_ctx(call_id=call_id)) is True
            run_after_tool_call_hooks(
                _tool_ctx(
                    call_id=call_id,
                    tool_result="blocked",
                    was_blocked=True,
                )
            )
            points = [record.interception_point.value for record in engine.records]
            assert points == ["pre_tool_call"]
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
                _tool_ctx(
                    tool_input={"q": "x"},
                    agent=agent,
                    crew=crew,
                    call_id="call-1",
                )
            )
            run_after_tool_call_hooks(
                _tool_ctx(
                    tool_input={"q": "x"},
                    tool_result="done",
                    agent=agent,
                    crew=crew,
                    call_id="call-1",
                )
            )
            pre_id = cap.at("pre_tool_call")[0]["tool_call"]["id"]
            post_id = cap.at("post_tool_call")[0]["tool_call"]["id"]
            assert pre_id == post_id
        finally:
            engine.close()

    def test_identical_tool_invocations_get_distinct_correlated_ids(self):
        """Repeated calls with equal arguments remain distinct audit events."""
        cap = _Capture()
        engine = use_agent_hooks(cap)
        crew = _crew("crew-1")
        agent = _agent("agent-A")
        try:
            for call_id, result in (("call-1", "first"), ("call-2", "second")):
                run_before_tool_call_hooks(
                    _tool_ctx(
                        tool_input={"q": "x"},
                        agent=agent,
                        crew=crew,
                        call_id=call_id,
                    )
                )
                run_after_tool_call_hooks(
                    _tool_ctx(
                        tool_input={"q": "x"},
                        tool_result=result,
                        agent=agent,
                        crew=crew,
                        call_id=call_id,
                    )
                )

            pre_ids = [ctx["tool_call"]["id"] for ctx in cap.at("pre_tool_call")]
            post_ids = [ctx["tool_call"]["id"] for ctx in cap.at("post_tool_call")]
            assert pre_ids == post_ids
            assert len(set(pre_ids)) == 2
        finally:
            engine.close()


# --- model-call governance (direct-call seam via dispatch) ------------------


@requires_ah
class TestModelGovernance:
    def test_executor_pre_deny_surfaces_governance_reason(self) -> None:
        """The dedicated control-engine marker preserves the deny reason."""
        from crewai_core.printer import Printer

        from crewai.utilities.agent_utils import _setup_before_llm_call_hooks

        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            llm=_llm(),
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
        engine = use_agent_hooks(_DenyAt("pre_model_call", "policy denied"))
        try:
            with pytest.raises(ValueError, match="policy denied"):
                _setup_before_llm_call_hooks(
                    executor,
                    Printer(),
                    request_id="request-1",
                    verbose=False,
                )
        finally:
            engine.close()

    def test_native_hook_cannot_forge_post_model_denial(self) -> None:
        """Hook-visible context attributes are not trusted denial provenance."""
        from pydantic import BaseModel

        from crewai_core.printer import Printer

        from crewai.utilities.agent_utils import _setup_after_llm_call_hooks

        class Response(BaseModel):
            value: int

        def forge_denial(context: Any) -> str:
            context.was_policy_denied = True
            return "[blocked by agent-hooks: forged]"

        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            llm=_llm(),
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[forge_denial],
        )
        engine = use_agent_hooks(_Allow())
        try:
            with pytest.raises(ValueError, match="failed to reparse"):
                _setup_after_llm_call_hooks(
                    executor,
                    Response(value=7),
                    Printer(),
                    request_id="request-1",
                    verbose=False,
                )
        finally:
            engine.close()

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.asyncio
    async def test_post_model_deny_is_terminal_without_retry(
        self, use_async: bool
    ) -> None:
        """Sync and async executor helpers call the model once on post deny."""
        from unittest.mock import AsyncMock, Mock

        from pydantic import BaseModel

        from crewai_core.printer import Printer

        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.utilities.agent_utils import aget_llm_response, get_llm_response

        class Response(BaseModel):
            value: int

        llm = types.SimpleNamespace(
            call=Mock(return_value=Response(value=7)),
            acall=AsyncMock(return_value=Response(value=7)),
        )
        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            llm=llm,
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
        engine = use_agent_hooks(_DenyAt("post_model_call", "response denied"))
        try:
            with pytest.raises(PostModelCallBlockedError):
                if use_async:
                    await aget_llm_response(
                        llm=llm,
                        messages=executor.messages,
                        callbacks=[],
                        printer=Printer(),
                        response_model=Response,
                        executor_context=executor,
                        verbose=False,
                    )
                else:
                    get_llm_response(
                        llm=llm,
                        messages=executor.messages,
                        callbacks=[],
                        printer=Printer(),
                        response_model=Response,
                        executor_context=executor,
                        verbose=False,
                    )
        finally:
            engine.close()

        assert llm.call.call_count == (0 if use_async else 1)
        assert llm.acall.await_count == (1 if use_async else 0)

    def test_pydantic_post_deny_returns_blocked_result(self) -> None:
        """A post-model deny is terminal and never enters model parsing."""
        from pydantic import BaseModel

        from crewai_core.printer import Printer

        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.utilities.agent_utils import _setup_after_llm_call_hooks

        class Response(BaseModel):
            value: int

        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            llm=_llm(),
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
        engine = use_agent_hooks(_DenyAt("post_model_call", "response denied"))
        try:
            with pytest.raises(PostModelCallBlockedError) as exc:
                _setup_after_llm_call_hooks(
                    executor,
                    Response(value=7),
                    Printer(),
                    request_id="request-1",
                    verbose=False,
                )
            assert exc.value.blocked_response == (
                "[blocked by agent-hooks: response denied]"
            )
        finally:
            engine.close()

    def test_invalid_pydantic_transform_fails_closed(self):
        """An invalid transformed model response never restores the original."""
        from pydantic import BaseModel

        from crewai_core.printer import Printer

        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.utilities.agent_utils import _setup_after_llm_call_hooks

        class Response(BaseModel):
            value: int

        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            llm=_llm(),
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
        engine = use_agent_hooks(_TransformAt("post_model_call", "not-json"))
        try:
            with pytest.raises(PostModelCallBlockedError) as exc:
                _setup_after_llm_call_hooks(
                    executor,
                    Response(value=7),
                    Printer(),
                    request_id="request-1",
                    verbose=False,
                )
            assert exc.value.failure_kind == "host_error"
            assert exc.value.request_id == "request-1"
            assert exc.value.reason.startswith("host_error:")
        finally:
            engine.close()

    def test_executor_model_call_ids_correlate(self):
        """Executor pre/post model records share one request identifier."""
        from crewai_core.printer import Printer

        from crewai.utilities.agent_utils import (
            _setup_after_llm_call_hooks,
            _setup_before_llm_call_hooks,
        )

        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            llm=_llm(),
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
        cap = _Capture()
        engine = use_agent_hooks(cap)
        try:
            request_id = "request-1"
            assert _setup_before_llm_call_hooks(
                executor, Printer(), request_id=request_id, verbose=False
            )
            assert (
                _setup_after_llm_call_hooks(
                    executor,
                    "model response",
                    Printer(),
                    request_id=request_id,
                    verbose=False,
                )
                == "model response"
            )
            pre = cap.at("pre_model_call")[0]
            post = cap.at("post_model_call")[0]
            assert pre["request_id"] == post["request_id"] == request_id
        finally:
            engine.close()

    @pytest.mark.asyncio
    async def test_async_pre_deny_prevents_file_processing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-model deny occurs before async file resolution or upload."""
        from crewai.llm import LLM

        processed = False

        async def fake_process(messages: Any) -> Any:
            nonlocal processed
            processed = True
            return messages

        llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)
        monkeypatch.setattr(llm, "_aprocess_message_files", fake_process)
        engine = use_agent_hooks(_DenyAt("pre_model_call", "blocked"))
        try:
            with pytest.raises(ValueError, match="blocked"):
                await llm.acall("hello")
            assert processed is False
        finally:
            engine.close()

    @pytest.mark.asyncio
    async def test_direct_async_llm_call_is_governed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct ``LLM.acall`` emits both model-call control points."""
        from crewai.llm import LLM

        async def fake_response(**kwargs: Any) -> str:
            return "model response"

        llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)
        monkeypatch.setattr(llm, "_ahandle_non_streaming_response", fake_response)
        cap = _Capture()
        engine = use_agent_hooks(cap)
        try:
            result = await llm.acall("hello")
            assert result == "model response"
            assert len(cap.at("pre_model_call")) == 1
            assert len(cap.at("post_model_call")) == 1
            pre = cap.at("pre_model_call")[0]
            post = cap.at("post_model_call")[0]
            assert pre["request_id"] == post["request_id"]
            assert pre["request_id"]
        finally:
            engine.close()

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.parametrize(
        "response",
        [
            "model response",
            {"content": "model response", "tool_calls": [], "finish_reason": "stop"},
            [
                types.SimpleNamespace(
                    id="call-1",
                    function=types.SimpleNamespace(name="shell", arguments="{}"),
                )
            ],
        ],
    )
    @pytest.mark.asyncio
    async def test_direct_post_model_deny_is_terminal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_async: bool,
        response: Any,
    ) -> None:
        """Direct responses cannot turn a governed block into ordinary output."""
        from unittest.mock import AsyncMock, Mock

        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.llm import LLM

        sync_response = Mock(return_value=response)
        async_response = AsyncMock(return_value=response)
        llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)
        monkeypatch.setattr(llm, "_handle_non_streaming_response", sync_response)
        monkeypatch.setattr(llm, "_ahandle_non_streaming_response", async_response)
        engine = use_agent_hooks(
            _DenyAt("post_model_call", "input is too long for policy")
        )
        try:
            with pytest.raises(PostModelCallBlockedError):
                if use_async:
                    await llm.acall("hello")
                else:
                    llm.call("hello")
        finally:
            engine.close()

        assert sync_response.call_count == (0 if use_async else 1)
        assert async_response.await_count == (1 if use_async else 0)

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.asyncio
    async def test_executor_dictionary_post_model_deny_is_terminal(
        self, use_async: bool
    ) -> None:
        """Executor dictionary responses cannot bypass post-model governance."""
        from unittest.mock import AsyncMock, Mock

        from crewai_core.printer import Printer

        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.utilities.agent_utils import aget_llm_response, get_llm_response

        response = {
            "content": "model response",
            "tool_calls": [],
            "finish_reason": "stop",
        }
        llm = types.SimpleNamespace(
            call=Mock(return_value=response),
            acall=AsyncMock(return_value=response),
        )
        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hello"}],
            llm=llm,
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
        engine = use_agent_hooks(_DenyAt("post_model_call", "response denied"))
        try:
            with pytest.raises(PostModelCallBlockedError):
                if use_async:
                    await aget_llm_response(
                        llm=llm,
                        messages=executor.messages,
                        callbacks=[],
                        printer=Printer(),
                        executor_context=executor,
                        verbose=False,
                    )
                else:
                    get_llm_response(
                        llm=llm,
                        messages=executor.messages,
                        callbacks=[],
                        printer=Printer(),
                        executor_context=executor,
                        verbose=False,
                    )
        finally:
            engine.close()

        assert llm.call.call_count == (0 if use_async else 1)
        assert llm.acall.await_count == (1 if use_async else 0)

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.asyncio
    async def test_direct_tool_call_metadata_can_drive_post_model_deny(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_async: bool,
    ) -> None:
        """Direct tool-call lists expose canonical names to model policy."""
        from unittest.mock import AsyncMock, Mock

        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.llm import LLM

        response = [
            types.SimpleNamespace(
                id="call-1",
                function=types.SimpleNamespace(
                    name="shell",
                    arguments='{"cmd": "echo safe"}',
                ),
            )
        ]
        sync_response = Mock(return_value=response)
        async_response = AsyncMock(return_value=response)
        llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)
        monkeypatch.setattr(llm, "_handle_non_streaming_response", sync_response)
        monkeypatch.setattr(llm, "_ahandle_non_streaming_response", async_response)
        engine = use_agent_hooks(_DenyToolNameAtPostModel("shell"))
        try:
            with pytest.raises(PostModelCallBlockedError):
                if use_async:
                    await llm.acall("hello")
                else:
                    llm.call("hello")
        finally:
            engine.close()

        assert sync_response.call_count == (0 if use_async else 1)
        assert async_response.await_count == (1 if use_async else 0)

    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.asyncio
    async def test_direct_post_model_deny_does_not_retry_unsupported_stop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_async: bool,
    ) -> None:
        """A policy reason cannot enter LiteLLM's unsupported-stop retry."""
        from unittest.mock import AsyncMock, Mock

        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.llm import LLM

        sync_response = Mock(
            side_effect=["model response", AssertionError("model retried")]
        )
        async_response = AsyncMock(
            side_effect=["model response", AssertionError("model retried")]
        )
        llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)
        monkeypatch.setattr(llm, "_handle_non_streaming_response", sync_response)
        monkeypatch.setattr(llm, "_ahandle_non_streaming_response", async_response)
        engine = use_agent_hooks(
            _DenyAt("post_model_call", "Unsupported parameter 'stop'")
        )
        try:
            with pytest.raises(PostModelCallBlockedError):
                if use_async:
                    await llm.acall("hello")
                else:
                    llm.call("hello")
        finally:
            engine.close()

        assert sync_response.call_count == (0 if use_async else 1)
        assert async_response.await_count == (1 if use_async else 0)

    def test_post_model_block_is_not_a_context_length_error(self) -> None:
        """Policy text cannot route a terminal block into summarization/retry."""
        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.utilities.agent_utils import is_context_length_exceeded

        error = PostModelCallBlockedError(
            "[blocked by agent-hooks: input is too long]",
            reason="input is too long",
            request_id="request-1",
            failure_kind="policy_denial",
        )

        assert is_context_length_exceeded(error) is False

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

    def test_invalid_model_transform_shape_blocks_call(self):
        engine = use_agent_hooks(_TransformAt("pre_model_call", {"invalid": True}))
        try:
            with pytest.raises(HookAborted):
                dispatch(
                    InterceptionPoint.PRE_MODEL_CALL,
                    _llm_ctx(),
                    reducer=before_llm_call_reducer,
                )
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

    def test_executor_structured_tool_calls_reach_governor(self):
        from crewai_core.printer import Printer

        from crewai.utilities.agent_utils import _setup_after_llm_call_hooks

        requested = types.SimpleNamespace(
            id="provider-call-1",
            function=types.SimpleNamespace(
                name="shell", arguments='{"cmd": "rm -rf /"}'
            ),
        )
        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            llm=_llm(),
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
        cap = _Capture()
        engine = use_agent_hooks(cap)
        try:
            answer = [requested]
            result = _setup_after_llm_call_hooks(
                executor,
                answer,
                Printer(),
                request_id="request-1",
                verbose=False,
            )
            assert result is answer
            seen = cap.at("post_model_call")
            assert seen[0]["response"]["tool_calls"] == [
                {
                    "id": "provider-call-1",
                    "name": "shell",
                    "args": {"cmd": "rm -rf /"},
                }
            ]
            assert seen[0]["request_id"] == "request-1"
        finally:
            engine.close()

    def test_executor_structured_tool_call_deny_prevents_execution(self):
        from crewai_core.printer import Printer

        from crewai.hooks.llm_hooks import PostModelCallBlockedError
        from crewai.utilities.agent_utils import _setup_after_llm_call_hooks

        requested = types.SimpleNamespace(
            id="provider-call-1",
            function=types.SimpleNamespace(name="shell", arguments="{}"),
        )
        executor = types.SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            llm=_llm(),
            iterations=0,
            agent=_agent(),
            task=None,
            crew=_crew(),
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
        engine = use_agent_hooks(_DenyAt("post_model_call", "tool denied"))
        try:
            with pytest.raises(PostModelCallBlockedError) as exc:
                _setup_after_llm_call_hooks(
                    executor,
                    [requested],
                    Printer(),
                    request_id="request-1",
                    verbose=False,
                )
            assert exc.value.blocked_response == (
                "[blocked by agent-hooks: tool denied]"
            )
        finally:
            engine.close()


# --- execution-boundary governance ------------------------------------------


@requires_ah
class TestBoundaryGovernance:
    def test_execution_end_abort_still_finalizes_governor(self) -> None:
        """Native end-hook aborts cannot skip shutdown or retain sessions."""
        from crewai.hooks.contexts import ExecutionEndContext

        def abort_end(context: Any) -> None:
            raise HookAborted(reason="native end denied")

        register(InterceptionPoint.EXECUTION_END, abort_end)
        engine = use_agent_hooks(_Allow())
        crew = _crew()
        try:
            for _ in range(3):
                dispatch(
                    InterceptionPoint.EXECUTION_START,
                    ExecutionStartContext(crew=crew, inputs={}, payload={}),
                )
                with pytest.raises(HookAborted, match="native end denied"):
                    dispatch(
                        InterceptionPoint.EXECUTION_END,
                        ExecutionEndContext(crew=crew),
                    )

            points = [record.interception_point.value for record in engine.records]
            assert points.count("agent_startup") == 3
            assert points.count("agent_shutdown") == 3
            assert engine._active_sessions == {}
            assert engine._builders == {}
        finally:
            engine.close()

    def test_execution_end_finalizes_session_when_builder_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A builder failure in the end hook must still finish the session."""
        from crewai.hooks.contexts import ExecutionEndContext

        engine = use_agent_hooks(_Allow())
        crew = _crew()
        try:
            dispatch(
                InterceptionPoint.EXECUTION_START,
                ExecutionStartContext(crew=crew, inputs={}, payload={}),
            )
            assert engine._active_sessions and engine._builders

            def boom(**_kwargs: Any) -> Any:
                raise RuntimeError("builder construction failed")

            monkeypatch.setattr(engine, "_builder", boom)
            with pytest.raises(HookAborted):
                dispatch(
                    InterceptionPoint.EXECUTION_END,
                    ExecutionEndContext(crew=crew),
                )

            assert engine._active_sessions == {}
            assert engine._builders == {}
        finally:
            engine.close()

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

    def test_output_transform_replaces_rich_crew_output(self):
        from crewai.crews.crew_output import CrewOutput
        from crewai.hooks.contexts import OutputContext
        from crewai.tasks.task_output import TaskOutput

        engine = use_agent_hooks(
            _TransformAt("output", "redacted", path="$target.content")
        )
        try:
            original = CrewOutput(
                raw="secret",
                json_dict={"secret": True},
                tasks_output=[
                    TaskOutput(
                        description="Return sensitive output",
                        raw="secret",
                        json_dict={"secret": True},
                        agent="tester",
                    )
                ],
            )
            ctx = OutputContext(crew=_crew(), output=original, payload=original)
            dispatch(InterceptionPoint.OUTPUT, ctx)
            assert isinstance(ctx.payload, CrewOutput)
            assert ctx.payload.raw == "redacted"
            assert ctx.payload.pydantic is None
            assert ctx.payload.json_dict is None
            assert ctx.payload.tasks_output[-1].raw == "redacted"
            assert ctx.payload.tasks_output[-1].pydantic is None
            assert ctx.payload.tasks_output[-1].json_dict is None
            assert original.tasks_output[-1].raw == "secret"
        finally:
            engine.close()

    def test_execution_start_deny_raises(self):
        engine = use_agent_hooks(_DenyAt("agent_startup"))
        try:
            ctx = ExecutionStartContext(crew=_crew(), inputs={}, payload={})
            with pytest.raises(HookAborted):
                dispatch(InterceptionPoint.EXECUTION_START, ctx)
            points = [record.interception_point.value for record in engine.records]
            assert points == ["agent_startup", "agent_shutdown"]
        finally:
            engine.close()

    def test_execution_end_failure_uses_schema_reason_and_does_not_raise(self):
        from crewai.hooks.contexts import ExecutionEndContext

        cap = _Capture()
        engine = use_agent_hooks(cap, _DenyAt("agent_shutdown"))
        crew = _crew()
        try:
            dispatch(
                InterceptionPoint.EXECUTION_START,
                ExecutionStartContext(crew=crew, inputs={}, payload={}),
            )
            dispatch(
                InterceptionPoint.EXECUTION_END,
                ExecutionEndContext(crew=crew, status="failed"),
            )
            shutdown = cap.at("agent_shutdown")[0]
            assert shutdown["summary"]["reason"] == "error"
        finally:
            engine.close()

    def test_repeated_runs_get_distinct_sessions(self):
        from crewai.hooks.contexts import ExecutionEndContext

        cap = _Capture()
        engine = use_agent_hooks(cap)
        crew = _crew("crew-1")
        try:
            for _ in range(2):
                dispatch(
                    InterceptionPoint.EXECUTION_START,
                    ExecutionStartContext(crew=crew, inputs={}, payload={}),
                )
                dispatch(
                    InterceptionPoint.EXECUTION_END,
                    ExecutionEndContext(crew=crew),
                )
            sessions = [ctx["session"]["id"] for ctx in cap.at("agent_startup")]
            assert len(set(sessions)) == 2
        finally:
            engine.close()

    def test_execution_start_matches_published_schema(self):
        """The emitted startup context conforms to the SDK's closed schema."""
        cap = _Capture()
        engine = use_agent_hooks(cap)
        try:
            dispatch(
                InterceptionPoint.EXECUTION_START,
                ExecutionStartContext(
                    crew=_crew(), inputs={"topic": "safe"}, payload={"topic": "safe"}
                ),
            )
            context = cap.at("agent_startup")[0]
            schema_path = files("agent_hooks").joinpath(
                "schema/agent-context/agent_startup.schema.json"
            )
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(context, schema)
        finally:
            engine.close()


# --- session & identity scoping ---------------------------------------------


@requires_ah
class TestSessionScoping:
    def test_concurrent_runs_of_one_owner_keep_distinct_sessions(self) -> None:
        """Each execution context resolves the session it started."""

        class Owner:
            id = "shared-owner"

        owner = Owner()
        engine = use_agent_hooks(_Allow())
        sessions_started = threading.Barrier(2)
        sessions_observed = threading.Barrier(2)
        observed: dict[str, tuple[str, str]] = {}

        def run(label: str) -> None:
            session_id = engine._begin_session(crew=owner)
            sessions_started.wait()
            observed[label] = (
                session_id,
                engine._current_session_id(crew=owner),
            )
            sessions_observed.wait()
            engine._finish_session(crew=owner)

        first = threading.Thread(target=run, args=("first",))
        second = threading.Thread(target=run, args=("second",))
        try:
            first.start()
            second.start()
            first.join(timeout=2.0)
            second.join(timeout=2.0)

            assert not first.is_alive()
            assert not second.is_alive()
            assert observed["first"][0] == observed["first"][1]
            assert observed["second"][0] == observed["second"][1]
            assert observed["first"][0] != observed["second"][0]
        finally:
            engine.close()

    def test_recycled_owner_address_does_not_reuse_stale_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A new owner cannot inherit a missing-end session at a reused address."""
        from crewai.hooks import agent_hooks_engine

        engine = use_agent_hooks(_Allow())
        stale_owner = _crew("stale-owner")
        current_owner = _crew("current-owner")
        recycled_address = 42
        monkeypatch.setattr(
            agent_hooks_engine,
            "id",
            lambda _owner: recycled_address,
            raising=False,
        )
        try:
            stale_session_id = engine._begin_session(crew=stale_owner)
            engine._builders[stale_session_id] = object()

            assert engine._current_session_id(crew=current_owner) == "current-owner"
            assert stale_session_id not in engine._builders
        finally:
            engine.close()

    def test_collected_owner_discards_missing_end_session(self) -> None:
        """Owner collection cleans session state when execution end is missing."""

        class Owner:
            id = "abandoned-owner"

        engine = use_agent_hooks(_Allow())
        owner = Owner()
        owner_id = id(owner)
        try:
            session_id = engine._begin_session(crew=owner)
            engine._builders[session_id] = object()

            del owner
            gc.collect()

            assert owner_id not in engine._active_sessions
            assert session_id not in engine._builders
        finally:
            engine.close()

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
    def test_host_failure_logs_exclude_exception_payloads(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Host failures log safe correlation only, never exception payloads."""
        import logging

        engine = use_agent_hooks(_Allow())
        ctx = _tool_ctx(
            crew=_crew("crew\nforged"),
            call_id="call\nforged",
        )
        try:
            def fail_run(
                _loop: Any,
                coro: Any,
                *_args: Any,
                **_kwargs: Any,
            ) -> Any:
                coro.close()
                raise RuntimeError("SECRET_MODEL_OUTPUT\nforged")

            monkeypatch.setattr(_EmitterLoop, "run", fail_run)
            with caplog.at_level(
                logging.ERROR,
                logger="crewai.hooks.agent_hooks_engine",
            ):
                assert run_before_tool_call_hooks(ctx) is True

            assert "SECRET_MODEL_OUTPUT" not in caplog.text
            assert "crew\\nforged" in caplog.text
            assert "call\\nforged" in caplog.text
        finally:
            engine.close()

    def test_adapter_failure_logs_exclude_exception_payloads(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Adapter failures also avoid exception text and traceback payloads."""
        import logging

        engine = use_agent_hooks(_Allow())

        def fail(context: Any) -> None:
            raise RuntimeError("SECRET_TOOL_INPUT\nforged")

        guarded = engine._wrap_fail_closed(InterceptionPoint.PRE_TOOL_CALL, fail)
        try:
            with caplog.at_level(
                logging.ERROR,
                logger="crewai.hooks.agent_hooks_engine",
            ):
                with pytest.raises(HookAborted):
                    guarded(_tool_ctx(call_id="call\nforged"))

            assert "SECRET_TOOL_INPUT" not in caplog.text
            assert "call\\nforged" in caplog.text
        finally:
            engine.close()

    def test_post_model_adapter_failure_survives_unmarkable_context(self) -> None:
        """A provenance storage failure cannot escape the blocked-result path."""

        class UnmarkableContext:
            __slots__ = ("response",)
            __hash__ = None

            def __init__(self) -> None:
                self.response = "model output"

        engine = use_agent_hooks(_Allow())

        def fail(context: Any) -> None:
            raise RuntimeError("adapter failed")

        guarded = engine._wrap_fail_closed(InterceptionPoint.POST_MODEL_CALL, fail)
        try:
            assert guarded(UnmarkableContext()) == (
                "[blocked by agent-hooks: host_error:engine_failed]"
            )
        finally:
            engine.close()

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

    def test_record_buffer_evicts_oldest_and_counts_drops(self) -> None:
        engine = use_agent_hooks(_Allow(), max_records=2)
        try:
            for call_id in ("call-1", "call-2", "call-3"):
                run_before_tool_call_hooks(_tool_ctx(call_id=call_id))

            assert len(engine.records) == 2
            assert engine.records_dropped == 1
            # Oldest-first eviction: the first emission (sequence 0) is dropped,
            # leaving the later two in ascending sequence order.
            assert [record.sequence for record in engine.records] == [1, 2]
            assert DEFAULT_MAX_RECORDS > 2
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
        assert active_engine() is None

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
