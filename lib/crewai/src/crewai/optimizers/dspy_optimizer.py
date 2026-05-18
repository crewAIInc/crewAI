"""DSPy-based prompt optimizer for CrewAI crews.

Requires crewai[dspy]: pip install 'crewai[dspy]'
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import inspect
import statistics
from typing import TYPE_CHECKING, Any, Literal

from crewai.optimizers.types import AgentInstructions, OptimizationResult


if TYPE_CHECKING:
    from crewai import Crew

# _dspy is typed as Any so all attribute accesses are valid regardless of dspy's stub state.
_dspy: Any = None

try:
    import dspy as _dspy  # type: ignore[import-not-found, no-redef]

    _HAS_DSPY = True
    _ModuleBase: type = _dspy.Module
except ImportError:
    _HAS_DSPY = False
    _ModuleBase = object


def _build_signature_for_agent(agent: Any) -> Any:
    """Build a dspy.Signature whose instructions encode only the agent's backstory.

    Only backstory is included so the optimized text writes back cleanly to
    agent.backstory without duplicating agent.goal (which the crew's prompt
    renderer already includes separately).
    """
    return _dspy.Signature(
        "task_input: str -> agent_output: str",
        instructions=agent.backstory,
    )


def _format_demos_as_messages(demos: list[Any]) -> list[dict[str, str]]:
    """Convert DSPy demo Examples to human/assistant message pairs for injection."""
    messages: list[dict[str, str]] = []
    for demo in demos:
        task_in = getattr(demo, "task_input", None)
        agent_out = getattr(demo, "agent_output", None)
        if task_in and agent_out:
            messages.append({"role": "user", "content": str(task_in)})
            messages.append({"role": "assistant", "content": str(agent_out)})
    return messages


def _get_example_inputs(example: Any) -> dict[str, Any]:
    """Extract input fields from a dspy.Example as a plain dict."""
    inputs_method = getattr(example, "inputs", None)
    if callable(inputs_method):
        # dspy.Example.inputs() returns another Example; dict() converts it
        return dict(inputs_method())
    # Fallback: if .inputs is already a dict attribute
    if isinstance(inputs_method, dict):
        return inputs_method
    return {}


class _CrewDSPyModule(_ModuleBase):  # type: ignore[misc]
    """Wraps a CrewAI Crew as a DSPy Module so teleprompters can optimize it."""

    def __init__(self, crew: Any) -> None:
        """Wrap crew and build one ChainOfThought predictor per agent."""
        super().__init__()
        self.crew = crew
        # One ChainOfThought predictor per agent; dict is found by dspy.Module.named_predictors()
        # NOTE: must NOT be named 'predictors' — that shadows dspy.Module.predictors() method
        self.agent_predictors: dict[str, Any] = {
            agent.role: _dspy.ChainOfThought(_build_signature_for_agent(agent))
            for agent in crew.agents
        }

    def forward(self, **inputs: Any) -> Any:
        """Run the crew and return a Prediction carrying the raw crew output."""
        crew_output = self.crew.kickoff(inputs=inputs)
        pred = _dspy.Prediction(final_output=str(crew_output))
        # Attach raw crew output so the metric adapter can forward it to the user metric
        pred._crew_output = crew_output
        return pred


def _select_teleprompter(
    algorithm: str,
    metric: Callable[..., float],
    num_trials: int,
    **kwargs: Any,
) -> Any:
    """Instantiate the DSPy teleprompter for the requested algorithm."""
    if algorithm == "MIPROv2":
        return _dspy.MIPROv2(metric=metric, num_candidates=num_trials, **kwargs)
    if algorithm == "BootstrapFewShot":
        return _dspy.BootstrapFewShot(metric=metric, **kwargs)
    if algorithm == "GEPA":
        if not hasattr(_dspy, "GEPA"):
            raise ImportError(
                "GEPA is not available in the installed version of dspy. "
                "Upgrade with: pip install 'dspy>=2.6'"
            )
        return _dspy.GEPA(metric=metric, **kwargs)
    raise ValueError(
        f"Unknown algorithm {algorithm!r}. "
        "Supported: 'MIPROv2', 'BootstrapFewShot', 'GEPA'"
    )


class DSPyOptimizer:
    """Optimize a CrewAI crew's agent instructions using DSPy teleprompters.

    Supports MIPROv2 (default), BootstrapFewShot, and GEPA algorithms.
    Requires crewai[dspy]: pip install 'crewai[dspy]'

    Example:
        optimizer = DSPyOptimizer(crew=my_crew, metric=my_metric)
        result = optimizer.compile(trainset=examples, num_trials=20)
        print(f"Score improved by {result.score_delta:.2f}")
    """

    def __init__(
        self,
        crew: Crew,
        metric: Callable[[Any, Any], float],
        algorithm: Literal["MIPROv2", "BootstrapFewShot", "GEPA"] = "MIPROv2",
        lm: Any | None = None,
    ) -> None:
        """Configure the optimizer with a crew, scoring metric, and algorithm choice."""
        if not _HAS_DSPY:
            raise ImportError(
                "crewai[dspy] is required. Install it with: pip install 'crewai[dspy]'"
            )
        self.crew = crew
        self.metric = metric
        self.algorithm = algorithm
        self.lm = lm  # None → use dspy's globally configured LM
        # List of (hook_type, hook_fn) tuples for cleanup in compile()'s finally block
        self._registered_hooks: list[tuple[str, Any]] = []
        self._compiled_module: _CrewDSPyModule | None = None

    def _make_before_hook(self) -> Callable[[Any], None]:
        """Build a before-LLM-call hook that injects compiled few-shot demos."""

        def before_hook(context: Any) -> None:
            """Inject compiled few-shot demos into the message list before each LLM call."""
            if self._compiled_module is None:
                return  # no compiled demos yet — pass through
            agent_role = getattr(context.agent, "role", None)
            if agent_role and agent_role in self._compiled_module.agent_predictors:
                cot = self._compiled_module.agent_predictors[agent_role]
                # In dspy 3.x, ChainOfThought wraps a Predict in .predict
                predictor = getattr(cot, "predict", cot)
                demos = getattr(predictor, "demos", [])
                if demos:
                    few_shot_messages = _format_demos_as_messages(demos)
                    # Always modify in-place — never reassign context.messages
                    context.messages.extend(few_shot_messages)

        return before_hook

    def _measure_score(self, trainset: list[Any]) -> float:
        """Run crew.kickoff() on every trainset example and return mean metric score."""
        scores: list[float] = []
        for example in trainset:
            inputs = _get_example_inputs(example)
            output = self.crew.kickoff(inputs=inputs)
            scores.append(float(self.metric(example, output)))
        return statistics.mean(scores) if scores else 0.0

    def compile(
        self,
        trainset: list[Any],
        num_trials: int = 20,
        **algorithm_kwargs: Any,
    ) -> OptimizationResult:
        """Run the DSPy optimization loop and write improved instructions back to agents.

        Args:
            trainset: Non-empty list of dspy.Example instances with .with_inputs() set.
            num_trials: Optimization budget — maps to MIPROv2 num_candidates (constructor)
                and num_trials (compile). Ignored by BootstrapFewShot.
            **algorithm_kwargs: Extra kwargs forwarded to the teleprompter constructor.

        Returns:
            OptimizationResult with the mutated crew and before/after metric scores.

        Raises:
            ValueError: If trainset is empty or metric is not callable.
            ImportError: If crewai[dspy] is not installed.
        """
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.optimizer_events import (
            OptimizationCompletedEvent,
            OptimizationFailedEvent,
            OptimizationStartedEvent,
        )
        from crewai.hooks.llm_hooks import (
            register_before_llm_call_hook,
            unregister_before_llm_call_hook,
        )

        # Reset state so a second compile() call never reads stale demos from the
        # previous run while the new baseline is being measured.
        self._compiled_module = None

        # ── 1. VALIDATE ──────────────────────────────────────────────────────
        if not trainset:
            raise ValueError(
                "trainset must be a non-empty list of dspy.Example instances"
            )
        if not callable(self.metric):
            raise TypeError("metric must be callable")

        # ── 4. REGISTER HOOKS (before try/finally so finally always runs cleanup) ─
        before_hook = self._make_before_hook()
        register_before_llm_call_hook(before_hook)
        self._registered_hooks.append(("before", before_hook))

        # Adapter: DSPy calls metric(example, prediction, trace=None) but the user
        # metric signature is metric(example, crew_output). We bridge via _crew_output.
        user_metric = self.metric

        def dspy_metric(example: Any, prediction: Any, trace: Any = None) -> float:
            """Bridge DSPy's (example, prediction, trace) signature to user's (example, crew_output)."""
            crew_output = getattr(prediction, "_crew_output", prediction)
            return float(user_metric(example, crew_output))

        try:
            # ── 2. MEASURE BASELINE ──────────────────────────────────────────
            baseline_score = self._measure_score(trainset)

            crewai_event_bus.emit(
                self,
                OptimizationStartedEvent(
                    crew_name=getattr(self.crew, "name", None),
                    algorithm=self.algorithm,
                    num_trials=num_trials,
                    trainset_size=len(trainset),
                ),
            )

            # ── 3. BUILD DSPy MODULE ─────────────────────────────────────────
            crew_module = _CrewDSPyModule(self.crew)
            # ── 5. COMPILE ───────────────────────────────────────────────────
            teleprompter = _select_teleprompter(
                self.algorithm, dspy_metric, num_trials, **algorithm_kwargs
            )

            # Temporarily override the DSPy LM if the user supplied one
            lm_ctx: Any = (
                _dspy.context(lm=self.lm)
                if self.lm is not None
                else contextlib.nullcontext()
            )

            # Pass num_trials to compile() only if the teleprompter accepts it
            # (dspy 3.x MIPROv2 accepts it; 2.x and BootstrapFewShot do not)
            compile_kwargs: dict[str, Any] = {}
            compile_sig = inspect.signature(teleprompter.compile)
            if "num_trials" in compile_sig.parameters:
                compile_kwargs["num_trials"] = num_trials

            with lm_ctx:
                compiled_module = teleprompter.compile(
                    crew_module, trainset=trainset, **compile_kwargs
                )

            self._compiled_module = compiled_module

            # ── 6. EXTRACT OPTIMIZED INSTRUCTIONS ────────────────────────────
            optimized_instructions: dict[str, AgentInstructions] = {}
            for agent in self.crew.agents:
                role = agent.role
                if role not in compiled_module.agent_predictors:
                    continue
                cot = compiled_module.agent_predictors[role]
                # In dspy 3.x ChainOfThought stores its Predict in .predict
                predictor = getattr(cot, "predict", cot)
                sig = getattr(predictor, "signature", None)
                new_instructions = getattr(sig, "instructions", "") if sig else ""

                if new_instructions:
                    optimized_instructions[role] = AgentInstructions(
                        backstory=new_instructions
                    )

            # ── 7. WRITE BACK (mutate agent fields in-place) ─────────────────
            for agent in self.crew.agents:
                instr = optimized_instructions.get(agent.role)
                if instr:
                    if instr.role:
                        agent.role = instr.role
                    if instr.goal:
                        agent.goal = instr.goal
                    if instr.backstory:
                        agent.backstory = instr.backstory

            # ── 8. MEASURE OPTIMIZED SCORE ────────────────────────────────────
            optimized_score = self._measure_score(trainset)

            # ── 9. BUILD RESULT + EMIT COMPLETED EVENT ────────────────────────
            result = OptimizationResult(
                crew=self.crew,
                baseline_score=baseline_score,
                optimized_score=optimized_score,
                optimized_instructions=optimized_instructions,
                num_trials=num_trials,
            )

            crewai_event_bus.emit(
                self,
                OptimizationCompletedEvent(
                    crew_name=getattr(self.crew, "name", None),
                    algorithm=self.algorithm,
                    baseline_score=baseline_score,
                    optimized_score=optimized_score,
                    score_delta=result.score_delta,
                    num_trials=num_trials,
                    version_id=result.version_id,
                ),
            )

            return result

        except Exception as exc:
            crewai_event_bus.emit(
                self,
                OptimizationFailedEvent(
                    crew_name=getattr(self.crew, "name", None),
                    error=str(exc),
                ),
            )
            raise

        finally:
            # ── 10. CLEANUP — always unregister hooks, even on exception ─────
            for hook_type, hook_fn in self._registered_hooks:
                if hook_type == "before":
                    unregister_before_llm_call_hook(hook_fn)
            self._registered_hooks.clear()
