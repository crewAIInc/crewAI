"""Velaris tools: run agent-written code in a box.

An agent that writes code needs somewhere safe to run it. These tools
compile a Velaris program, report what it can touch, and run it under
an effect budget the crew's author chooses - not the agent. A program
granted only "io" cannot read files, reach the network or call Python,
whatever its source claims, and a refusal cannot be caught by the
program.

    from crewai_tools import VelarisAuditTool, VelarisRunTool

    agent = Agent(
        role="Data analyst",
        tools=[VelarisAuditTool(), VelarisRunTool(allow=["io"])],
        ...
    )

The program runs in a separate, killable process bounded by ``timeout``
(seconds, default 30) and ``max_memory_mb`` (default 512), so a program
that never ends or eats memory is stopped and the crew's worker survives
it. The memory cap is enforced on Linux and macOS; on Windows it is
recorded but not enforced. Both limits need velaris-lang 2.59.0 or
newer; with an older compiler the program runs unbounded.

Not a security boundary: allow=["ffi"] grants everything Python can
do. It is a guard for the ordinary case of running a script a model
wrote.

Install the compiler once: pip install velaris-lang z3-solver
The z3-solver is optional; without it promises are checked while
running rather than proven beforehand, and the audit says so.
"""

import importlib
import inspect
import json
from types import ModuleType
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


def _import_velaris() -> ModuleType:
    """Import the compiler lazily so the base install stays light.

    Returns:
        The ``velaris`` module.

    Raises:
        ImportError: If ``velaris-lang`` is not installed.
    """
    try:
        return importlib.import_module("velaris")
    except ImportError as exc:
        raise ImportError(
            "The 'velaris-lang' package is required for Velaris tools. "
            "Install it with: uv add crewai-tools --extra velaris  (or) "
            "pip install velaris-lang"
        ) from exc


def _supports_limits(velaris: ModuleType) -> bool:
    """Report whether the installed compiler accepts run-time limits.

    ``timeout`` and ``max_memory_mb`` arrived in velaris-lang 2.59.0.

    Args:
        velaris: The imported ``velaris`` module.

    Returns:
        True if ``velaris.run`` accepts ``timeout`` and ``max_memory_mb``.
    """
    params = inspect.signature(velaris.run).parameters
    return "timeout" in params and "max_memory_mb" in params


class VelarisAuditToolSchema(BaseModel):
    """Input for VelarisAuditTool."""

    source: str = Field(..., description="The Velaris program to inspect.")


class VelarisRunToolSchema(BaseModel):
    """Input for VelarisRunTool."""

    source: str = Field(..., description="The Velaris program to run.")
    stdin: str = Field(
        default="", description="Text to feed the program on standard input."
    )
    args: list[str] = Field(
        default_factory=list, description="Command-line arguments for the program."
    )


class VelarisCardTool(BaseTool):
    """The Velaris language, small enough to read before writing it."""

    name: str = "Velaris language card"
    description: str = (
        "Read the Velaris language card (about 2,300 words) before "
        "writing Velaris. It covers syntax, the rules models get wrong, "
        "every builtin with its effects, and the error table."
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["velaris-lang"])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _import_velaris()

    def _run(self, **_: Any) -> str:
        """Return the language card.

        Returns:
            The Velaris language card as text.
        """
        card: str = _import_velaris().card()
        return card


class VelarisAuditTool(BaseTool):
    """What a program can touch, promise and fail at - before running."""

    name: str = "Audit a Velaris program"
    description: str = (
        "Audit a Velaris program before running it: which effects it "
        "can perform (io, fs, net, clock, rand, ffi), what each function "
        "promises and whether that was proven before running, what can "
        "fail, and the command to run it safely. Returns JSON in the "
        "velaris.audit/1 format."
    )
    args_schema: type[BaseModel] = VelarisAuditToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["velaris-lang"])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _import_velaris()

    def _run(self, source: str) -> str:
        """Audit ``source`` without running it.

        Args:
            source: The Velaris program to inspect.

        Returns:
            The audit report as JSON in the ``velaris.audit/1`` format.
        """
        velaris = _import_velaris()
        return json.dumps(velaris.audit(source).as_dict(), indent=2)


class VelarisRunTool(BaseTool):
    """Run a program under an effect budget the crew's author chose.

    The program runs in a separate, killable process bounded by ``timeout``
    and ``max_memory_mb``, so a program that never ends or eats memory is
    stopped and reported as such. The memory cap is enforced on Linux and
    macOS only; the timeout is enforced everywhere. Both need velaris-lang
    2.59.0 or newer.

    Args:
        allow: The effects the program may perform, chosen from io, fs, net,
            clock, rand and ffi. Defaults to ``["io"]``: the program can print
            and nothing else.
        timeout: Seconds the program may run before it is stopped.
        max_memory_mb: Memory the program may use, in MB, before it is stopped.
        **kwargs: Additional keyword arguments passed to BaseTool.

    Example:
        >>> tool = VelarisRunTool(allow=["io"], timeout=10)
        >>> tool.run(source="fn main() uses io { print(6 * 7) }")
    """

    name: str = "Run a Velaris program"
    description: str = (
        "Run a Velaris program in a sandbox. Effects outside the budget "
        "this tool was created with are refused while the program runs, "
        "whatever the source claims, and a refusal cannot be caught. "
        "A program that runs too long or uses too much memory is stopped. "
        "Returns the program's output, or the problem that stopped it."
    )
    args_schema: type[BaseModel] = VelarisRunToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["velaris-lang"])
    allow: list[str] = Field(
        default_factory=lambda: ["io"],
        description="The effects the program may perform: io, fs, net, clock, rand, ffi.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Seconds the program may run before it is stopped.",
    )
    max_memory_mb: int = Field(
        default=512,
        gt=0,
        description="Memory cap in MB, enforced on Linux and macOS.",
    )

    def __init__(
        self,
        allow: list[str] | None = None,
        timeout: float | None = None,
        max_memory_mb: int | None = None,
        **kwargs: Any,
    ) -> None:
        if allow is not None:
            kwargs["allow"] = list(allow)
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_memory_mb is not None:
            kwargs["max_memory_mb"] = max_memory_mb
        super().__init__(**kwargs)
        _import_velaris()

    def _run(
        self,
        source: str,
        stdin: str = "",
        args: list[str] | None = None,
    ) -> str:
        """Run ``source`` under this tool's effect budget and limits.

        Args:
            source: The Velaris program to run.
            stdin: Text to feed the program on standard input.
            args: Command-line arguments for the program.

        Returns:
            The program's output, or a description of what stopped it.
        """
        velaris = _import_velaris()
        limits: dict[str, Any] = {}
        if _supports_limits(velaris):
            # A separate, killable process: a program that never ends or
            # eats memory is stopped, and the crew's worker survives it.
            limits = {"timeout": self.timeout, "max_memory_mb": self.max_memory_mb}
        result = velaris.run(
            source, allow=set(self.allow), stdin=stdin, args=args or [], **limits
        )
        if result.ok:
            output: str = result.output or "(the program printed nothing)"
            return output
        lines: list[str] = []
        if getattr(result, "timed_out", False):
            lines.append(
                f"STOPPED: the program ran longer than {self.timeout} seconds."
            )
        elif getattr(result, "out_of_memory", False):
            lines.append(
                f"STOPPED: the program used more than {self.max_memory_mb} MB."
            )
        if result.refused_effect:
            lines.append(
                f"REFUSED: the program tried to use "
                f"'{result.refused_effect}', which this tool does not "
                f"allow (it allows: {', '.join(sorted(self.allow))})."
            )
        for p in result.problems:
            lines.append(f"line {p.line}: [{p.code}] {p.message}")
            lines.extend(f"    fix: {fix}" for fix in (p.fixes or [])[:2])
        if result.output:
            lines.append("output before it stopped:")
            lines.append(result.output)
        return "\n".join(lines) or "the program failed without a message"
