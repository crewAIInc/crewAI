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
it. The limits are enforced on every supported compiler version: on
velaris-lang 2.59.0 and newer by ``velaris.run`` itself, on older
versions by the tool's own subprocess. The memory cap is enforced on
Linux; best-effort on macOS; not applied on Windows.

Not a security boundary: allow=["ffi"] grants everything Python can
do. It is a guard for the ordinary case of running a script a model
wrote.

Install the compiler once: pip install velaris-lang z3-solver
The z3-solver is optional; without it promises are checked while
running rather than proven beforehand, and the audit says so.
"""

from collections.abc import Callable
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


_REFUSED_EFFECT = re.compile(r"needs the '(\w+)' effect")
_PROBLEM_LINE = re.compile(r"line (\d+)")


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


def _memory_limiter(max_memory_mb: int) -> Callable[[], None] | None:
    """Build a ``preexec_fn`` that caps the child's address space.

    Args:
        max_memory_mb: The cap in megabytes.

    Returns:
        A function to run in the child before exec, or None on Windows,
        where address-space limits are not available. The limit is
        reliably honoured on Linux and best-effort on macOS.
    """
    if sys.platform == "win32":
        return None
    import resource

    limit = max_memory_mb * 1024 * 1024

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    return _apply


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
    stopped and reported as such. On velaris-lang 2.59.0 and newer the
    limits are enforced by ``velaris.run``; on older versions the tool runs
    the compiler as its own bounded subprocess. The memory cap is enforced
    on Linux; best-effort on macOS; not applied on Windows. The timeout is
    enforced everywhere.

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
        description=(
            "Memory cap in MB: enforced on Linux; best-effort on macOS; "
            "not applied on Windows."
        ),
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
        if not _supports_limits(velaris):
            return self._run_in_subprocess(source, stdin, args or [])
        # A separate, killable process: a program that never ends or
        # eats memory is stopped, and the crew's worker survives it.
        result = velaris.run(
            source,
            allow=set(self.allow),
            stdin=stdin,
            args=args or [],
            timeout=self.timeout,
            max_memory_mb=self.max_memory_mb,
        )
        if result.ok:
            output: str = result.output or "(the program printed nothing)"
            return output
        lines: list[str] = []
        if result.timed_out:
            lines.append(self._timed_out())
        elif result.out_of_memory:
            lines.append(self._out_of_memory())
        if result.refused_effect:
            lines.append(self._refused(result.refused_effect))
        for p in result.problems:
            lines.append(f"line {p.line}: [{p.code}] {p.message}")
            lines.extend(f"    fix: {fix}" for fix in (p.fixes or [])[:2])
        return self._finish(lines, result.output)

    def _run_in_subprocess(self, source: str, stdin: str, args: list[str]) -> str:
        """Run the compiler as a bounded subprocess.

        Used when the installed compiler predates ``velaris.run``'s own
        limits (velaris-lang < 2.59.0). The outcome is mapped to the same
        report the ``velaris.run`` path produces. One difference remains on
        this path: the compiler's command line puts everything after the
        file name into the program's ``args()``, so the program also sees
        the ``--allow`` flag and its value ahead of ``args``.

        Args:
            source: The Velaris program to run.
            stdin: Text to feed the program on standard input.
            args: Command-line arguments for the program.

        Returns:
            The program's output, or a description of what stopped it.
        """
        fd, path = tempfile.mkstemp(suffix=".vel", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(source)
            command = [
                sys.executable,
                "-m",
                "velaris",
                path,
                "--allow",
                ",".join(sorted(self.allow)),
                *args,
            ]
            try:
                completed = subprocess.run(  # noqa: S603
                    command,
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    preexec_fn=_memory_limiter(self.max_memory_mb),
                )
            except subprocess.TimeoutExpired as exc:
                # Typed as bytes in the stubs, but str at runtime with text=True.
                raw: bytes | str | None = exc.stdout
                partial = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, bytes)
                    else raw or ""
                )
                return self._finish([self._timed_out()], partial)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        if completed.returncode == 0:
            return completed.stdout or "(the program printed nothing)"

        stderr = completed.stderr or ""
        lines: list[str] = []
        if "error[E310]" in stderr:
            effect_match = _REFUSED_EFFECT.search(stderr)
            effect = effect_match.group(1) if effect_match else "an effect"
            lines.append(self._refused(effect))
            stderr_lines = [line for line in stderr.splitlines() if line.strip()]
            line_match = _PROBLEM_LINE.search(stderr)
            line_no = line_match.group(1) if line_match else "0"
            message = stderr_lines[0].replace("error[E310] ", "", 1)
            lines.append(f"line {line_no}: [E310] {message}")
        elif "MemoryError" in stderr or completed.returncode in (-9, 137):
            lines.append(self._out_of_memory())
        else:
            first = next((line for line in stderr.splitlines() if line.strip()), "")
            lines.append(
                first or f"the program exited with code {completed.returncode}"
            )
        return self._finish(lines, completed.stdout)

    def _timed_out(self) -> str:
        """Describe a run stopped by the time limit.

        Returns:
            The STOPPED line for a timeout.
        """
        return f"STOPPED: the program ran longer than {self.timeout} seconds."

    def _out_of_memory(self) -> str:
        """Describe a run stopped by the memory cap.

        Returns:
            The STOPPED line for the memory cap.
        """
        return f"STOPPED: the program used more than {self.max_memory_mb} MB."

    def _refused(self, effect: str) -> str:
        """Describe a refused effect.

        Args:
            effect: The effect the program tried to use.

        Returns:
            The REFUSED line naming the effect and the budget.
        """
        return (
            f"REFUSED: the program tried to use '{effect}', which this tool "
            f"does not allow (it allows: {', '.join(sorted(self.allow))})."
        )

    @staticmethod
    def _finish(lines: list[str], output: str | None) -> str:
        """Append any partial output and join the report.

        Args:
            lines: The report lines gathered so far.
            output: What the program printed before it stopped, if anything.

        Returns:
            The joined report, or a fallback when there is nothing to say.
        """
        if output:
            lines.append("output before it stopped:")
            lines.append(output)
        return "\n".join(lines) or "the program failed without a message"
