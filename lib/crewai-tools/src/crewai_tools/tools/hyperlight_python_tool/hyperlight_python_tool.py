from __future__ import annotations

from builtins import type as type_
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import IO, Any

from crewai.tools import BaseTool
from crewai.types.callback import SerializableCallable
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import Self


# The first hluk release whose console carries lines longer than 4 KiB whole
MIN_HLUK_VERSION = (0, 18, 0)

# The longest timeout: about 11 days, well within what a wait takes everywhere
_MAX_SECONDS = 1_000_000

# Most of a startup failure's output kept in its error, from the end
_STARTUP_ERROR_CHARS = 4000

# Where the published runtime images live, as hluk names them
_HLUK_REGISTRY = "ghcr.io/hyperlight-dev/hyperlight-unikraft/"

# What an internal command (`--version`, the warm-up) keeps of its output,
# whatever a call's max_output_bytes
_INTERNAL_OUTPUT_BYTES = 64 * 1024

# How long the output is still read once hluk is gone: something it left
# running could hold its pipes open for good
_DRAIN_GRACE_SECONDS = 1.0

_UTF8_CONTINUATION = bytes(range(0x80, 0xC0))


def _is_agent_image(runtime: str) -> bool:
    """Whether `runtime` is the published `agent` image, by name or reference."""
    if runtime == "agent":
        return True
    if not runtime.startswith(_HLUK_REGISTRY):
        return False
    name = runtime[len(_HLUK_REGISTRY) :].split("@", 1)[0].split(":", 1)[0]
    return name == "agent"


def _format_seconds(value: float) -> str:
    """Render a duration with a correctly pluralized unit, e.g. `1 second`."""
    number = f"{value:.3f}".rstrip("0").rstrip(".")
    if number == "0":
        number = f"{value:g}"
    return f"{number} second" if number == "1" else f"{number} seconds"


# The sentences of the generated description
_INTRO = (
    "Execute a self-contained Python script inside an isolated Hyperlight "
    "micro-VM and return its exit code, stdout, and stderr. Each call starts "
    "from a fresh sandbox: variables and imports from earlier calls are gone, "
    "so print every result you need."
)
_PACKAGES = (
    "numpy, pandas, scipy, scikit-learn, matplotlib, pillow, and requests are "
    "available."
)
_NO_NETWORK = "The sandbox has no network access."
_NETWORK = "The sandbox can connect only to: {hosts}."
# hluk_args can open the network beyond what net_allow says
_NETWORK_BY_ARGS = "The sandbox may have network access."
_LIMIT = (
    "Code that runs longer than {limit} is stopped. Use this for calculations, "
    "data processing, or any code that should not run on the host."
)

# Any description these sentences make, whatever the configuration it was made
# for: one dumped and validated again with other settings is still generated
# Placeholders only take what the configuration can put there (a list of hosts,
# a number of seconds), so that a sentence the caller adds makes it theirs
_HOSTS_PATTERN = r"[^\s,]+(?:, [^\s,]+)*"
_LIMIT_PATTERN = r"[0-9.e+-]+ seconds?"
_GENERATED_DESCRIPTION = re.compile(
    re.escape(_INTRO)
    + f"( {re.escape(_PACKAGES)})?"
    + f" ({re.escape(_NO_NETWORK)}|{re.escape(_NETWORK_BY_ARGS)}|"
    + re.escape(_NETWORK).replace(re.escape("{hosts}"), _HOSTS_PATTERN)
    + ") "
    + re.escape(_LIMIT).replace(re.escape("{limit}"), _LIMIT_PATTERN)
)


def _build_description(
    runtime: str, net_allow: tuple[str, ...], timeout: float, hluk_args: tuple[str, ...]
) -> str:
    """Build the LLM-facing description from what the sandbox offers."""
    parts = [_INTRO]
    if _is_agent_image(runtime):
        parts.append(_PACKAGES)
    # --port lets the guest accept connections, even with no egress
    if any(
        arg in ("--net", "--port") or arg.startswith(("--net-", "--net=", "--port="))
        for arg in hluk_args
    ):
        parts.append(_NETWORK_BY_ARGS)
    elif net_allow:
        parts.append(_NETWORK.format(hosts=", ".join(net_allow)))
    else:
        parts.append(_NO_NETWORK)
    parts.append(_LIMIT.format(limit=_format_seconds(timeout)))
    return " ".join(parts)


def _complete_utf8(data: bytes) -> bytes:
    """Drop a multi-byte character that the end of `data` cuts short."""
    lead = len(data.rstrip(_UTF8_CONTINUATION)) - 1
    if lead < 0 or data[lead] < 0xC0:
        return data
    length = 2 if data[lead] < 0xE0 else 3 if data[lead] < 0xF0 else 4
    return data if len(data) - lead >= length else data[:lead]


def _check_env_name(name: str) -> None:
    # hluk splits KEY=VALUE at the first "="
    if not name or "=" in name or "\0" in name:
        raise ValueError(f"not an environment variable name: {name!r}")


def _never_cache(_args: Any = None, _result: Any = None) -> bool:
    """Refuse caching: the same code can give another result (randomness, the
    network, a timeout), and a cache hit would not run it at all."""
    return False


class HyperlightPythonToolSchema(BaseModel):
    code: str = Field(
        ...,
        description="Python source to execute inside the sandbox.",
    )


class HyperlightPythonTool(BaseTool):
    """Run Python code in a local Hyperlight micro-VM through the `hluk` CLI.

    Each call runs in a fresh, hardware-isolated micro-VM restored from a warm
    snapshot of the runtime image, so nothing carries over between calls and
    nothing the code does reaches the host beyond what the tool allows. The
    guest has no network access unless hosts are listed in `net_allow`.

    Needs the `hluk` binary from
    https://github.com/hyperlight-dev/hyperlight-unikraft and a hypervisor:
    KVM on Linux, WHP on Windows, Hypervisor.framework on macOS.

    The generated description tells the model what the sandbox offers, and is
    kept in sync with `runtime`, `net_allow`, `timeout` and `hluk_args`: so is any
    description in the generated form, such as one dumped and validated again.
    Any other description supplied by the caller is left alone.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: str = "Hyperlight Sandbox Python"
    description: str = _build_description("agent", (), 60.0, ())
    args_schema: type_[BaseModel] = HyperlightPythonToolSchema
    cache_function: SerializableCallable = Field(
        default=_never_cache,
        description="Results are never cached: running the same code again can give another result.",
    )

    hluk_path: str | None = Field(
        default=None,
        description="Path to the `hluk` binary. Defaults to the `hluk` found on PATH.",
    )
    runtime: str = Field(
        default="agent",
        description=(
            "Published runtime image to run, by name or full image reference. "
            "The `agent` image ships CPython 3.12 with numpy, pandas, scipy, "
            "scikit-learn, matplotlib, pillow, and requests."
        ),
    )
    # A tuple, so that it only changes by assignment, which keeps the
    # description in sync
    net_allow: tuple[str, ...] = Field(
        default=(),
        description=(
            "Hosts or IPs the guest may connect to. Defaults to none: every "
            "connection is refused."
        ),
    )
    envs: dict[str, str] | None = Field(
        default=None,
        description=(
            "Environment variables to set inside the sandbox. They are passed on "
            "the `hluk` command line, which other processes on the host can "
            "read, so do not put secrets here."
        ),
    )
    scratch_mb: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Guest memory in MiB. Defaults to the size the runtime image is "
            "tested with."
        ),
    )
    # A tuple, like net_allow, as it shapes the description too
    hluk_args: tuple[str, ...] = Field(
        default=(),
        description=(
            "Additional arguments to pass to `hluk run`, e.g. "
            '["--mount", "/data:/data:ro"].'
        ),
    )
    timeout: float = Field(
        default=60.0,
        gt=0,
        # Windows waits take whole milliseconds in 32 bits
        le=_MAX_SECONDS,
        allow_inf_nan=False,
        description="Maximum seconds a call's code may run before the sandbox is stopped.",
    )
    startup_timeout: float = Field(
        default=300.0,
        gt=0,
        # Windows waits take whole milliseconds in 32 bits
        le=_MAX_SECONDS,
        allow_inf_nan=False,
        description=(
            "Maximum seconds to wait for the first call to prepare the sandbox: "
            "it pulls the runtime image and saves its warm snapshot."
        ),
    )
    max_output_bytes: int = Field(
        default=1024 * 1024,
        gt=0,
        description=(
            "Bytes of stdout and of stderr kept per call. Longer output keeps "
            "its beginning and end, with a note of how much was left out."
        ),
    )

    # The command the warm snapshot was prepared for, so that a change to the
    # configuration prepares it again rather than under a call's timeout
    _prepared: tuple[str, ...] | None = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    # The hluk binary whose version was checked
    _checked_hluk: str | None = PrivateAttr(default=None)

    @field_validator("net_allow")
    @classmethod
    def _check_hosts(cls, hosts: tuple[str, ...]) -> tuple[str, ...]:
        for host in hosts:
            if not host or any(c.isspace() or c == "," for c in host):
                raise ValueError(f"not a host name or IP: {host!r}")
        return hosts

    @field_validator("envs")
    @classmethod
    def _check_env_names(cls, envs: dict[str, str] | None) -> dict[str, str] | None:
        for name in envs or {}:
            _check_env_name(name)
        return envs

    @model_validator(mode="after")
    def _sync_description(self) -> Self:
        # The default is in the generated form too; a subclass's own is not
        if _GENERATED_DESCRIPTION.fullmatch(self.description):
            # Bypass the field setter: assigning here would re-enter validation
            self.__dict__["description"] = _build_description(
                self.runtime, self.net_allow, self.timeout, self.hluk_args
            )
        return self

    def _run(self, code: str) -> dict[str, Any]:
        command = self._prepare()
        with tempfile.TemporaryDirectory(prefix="crewai-hyperlight-") as workdir:
            script = os.path.join(workdir, "main.py")
            # A lone surrogate, which a model's JSON can carry, is written as
            # its escape rather than failing to encode: the same character in
            # a string literal, not in a raw string
            # newline="": the code byte for byte, rather than \r\r\n on Windows
            with open(
                script, "w", encoding="utf-8", errors="backslashreplace", newline=""
            ) as f:
                f.write(code)
            result = self._execute([*command, script], self.timeout)
        if result["timed_out"]:
            # Prepare again before the next call, under startup_timeout: if hluk
            # was upgraded or its cache cleared, this call spent its timeout on a
            # pull. Without the lock, which a warm-up holds for minutes: a plain
            # attribute write is atomic.
            self._prepared = None
        return result

    def _prepare(self) -> list[str]:
        """Return the `hluk run` command, preparing its warm snapshot first.

        The first run of a configuration pulls the runtime image and saves its
        warm snapshot, which can take minutes, so it gets `startup_timeout`
        rather than eating into the call's `timeout`.
        """
        command = self._base_command(self._find_hluk())
        with self._lock:
            if self._prepared == tuple(command):
                return command
            if self._checked_hluk != command[0]:
                self._check_version(command[0])
                self._checked_hluk = command[0]
            result = self._execute(
                [*command, "--exec", "pass"],
                self.startup_timeout,
                _INTERNAL_OUTPUT_BYTES,
            )
            if result["exit_code"] != 0:
                output = "\n".join(
                    part.strip()
                    for part in (result["stdout"], result["stderr"])
                    if part.strip()
                )
                # A timeout says how long in the output already
                status = (
                    "timed out"
                    if result["timed_out"]
                    else f"exit status {result['exit_code']}"
                )
                raise RuntimeError(
                    f"Failed to start the Hyperlight sandbox ({status}): "
                    f"{output[-_STARTUP_ERROR_CHARS:] or 'no output'}"
                )
            self._prepared = tuple(command)
        return command

    def _find_hluk(self) -> str:
        if self.hluk_path is None:
            hluk = shutil.which("hluk")
        else:
            hluk = shutil.which(os.path.expanduser(self.hluk_path))
        if hluk is not None:
            # Absolute, so that a later chdir cannot swap the binary that the
            # version check and the warm-up were for
            return os.path.abspath(hluk)
        # A bare name is looked up on PATH: a file of that name here is not it
        path = os.path.expanduser(self.hluk_path or "")
        has_directory = os.sep in path or (os.altsep or os.sep) in path
        if has_directory and os.path.isfile(path):
            raise RuntimeError(
                f"{self.hluk_path!r} is not executable: make it so with `chmod +x`."
            )
        where = "on PATH" if self.hluk_path is None else f"at {self.hluk_path!r}"
        raise RuntimeError(
            f"Could not find the `hluk` binary {where}: install it from "
            "https://github.com/hyperlight-dev/hyperlight-unikraft or pass "
            "its location with `hluk_path`."
        )

    def _check_version(self, hluk: str) -> None:
        try:
            result = self._execute([hluk, "--version"], 30, _INTERNAL_OUTPUT_BYTES)
        except OSError as exc:
            raise RuntimeError(f"Could not run `{hluk} --version`: {exc}") from exc
        if result["timed_out"]:
            raise RuntimeError(f"`{hluk} --version` did not answer within 30 seconds.")
        if result["exit_code"] != 0:
            # It cannot run at all (a missing library, say), which a version
            # check would misreport as too old
            raise RuntimeError(
                f"`{hluk} --version` failed: "
                f"{result['stderr'].strip()[-_STARTUP_ERROR_CHARS:] or 'no output'}"
            )
        version = result["stdout"]
        match = re.search(r"(\d+)\.(\d+)\.(\d+)(-\S+)?", version)
        if match is not None:
            release = tuple(int(part) for part in match.groups()[:3])
            # A pre-release (0.18.0-rc1) comes before its release
            is_pre_release = match.group(4) is not None
        if (
            match is None
            or release < MIN_HLUK_VERSION
            or (release == MIN_HLUK_VERSION and is_pre_release)
        ):
            minimum = ".".join(str(part) for part in MIN_HLUK_VERSION)
            raise RuntimeError(
                f"HyperlightPythonTool needs hluk v{minimum} or later, but "
                f"{hluk} reports {version.strip()!r}."
            )

    def _base_command(self, hluk: str) -> list[str]:
        command = [hluk, "run", "--runtime", self.runtime]
        for host in self.net_allow:
            command += ["--net-allow", host]
        for key, value in (self.envs or {}).items():
            # Again here: the dict can change in place, past the validator
            _check_env_name(key)
            command += ["--env", f"{key}={value}"]
        if self.scratch_mb is not None:
            command += ["--scratch-mb", str(self.scratch_mb)]
        return [*command, *self.hluk_args]

    def _execute(
        self, command: list[str], timeout: float, limit: int | None = None
    ) -> dict[str, Any]:
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        captures = [_Capture(limit or self.max_output_bytes) for _ in range(2)]
        readers = [
            threading.Thread(target=capture.drain, args=(stream,), daemon=True)
            for capture, stream in zip(
                captures, (process.stdout, process.stderr), strict=True
            )
        ]
        timed_out = False
        try:
            # In here, so that a thread that fails to start does not leave the
            # micro-VM running
            for reader in readers:
                reader.start()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also on an interrupt: the micro-VM must not outlive the call
            if process.poll() is None:
                process.kill()
                process.wait()
            deadline = time.monotonic() + _DRAIN_GRACE_SECONDS
            for reader in readers:
                if reader.is_alive():
                    reader.join(timeout=max(0.0, deadline - time.monotonic()))
        stdout, stderr = (capture.text() for capture in captures)
        if timed_out:
            stderr += (
                f"\nTimed out after {_format_seconds(timeout)}; the sandbox was "
                "stopped."
            )
        return {
            "exit_code": None if timed_out else process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
        }


class _Capture:
    """One of hluk's output streams, read to the end, keeping its first and
    last limit/2 bytes: a traceback comes last."""

    def __init__(self, limit: int) -> None:
        self._half = limit // 2
        self._tail_limit = limit - self._half
        self._head, self._tail, self._dropped = bytearray(), bytearray(), 0
        # The reader may still be adding when the call ends
        self._lock = threading.Lock()

    def drain(self, stream: IO[bytes] | None) -> None:
        # Reads to the end, so that a full pipe never blocks the guest
        if stream is None:
            return
        with stream:
            # As it arrives rather than a whole buffer at a time, so that what
            # is kept is current when the call ends
            while chunk := os.read(stream.fileno(), 65536):
                with self._lock:
                    taken = max(0, min(len(chunk), self._half - len(self._head)))
                    self._head += chunk[:taken]
                    self._tail += chunk[taken:]
                    excess = len(self._tail) - self._tail_limit
                    if excess > 0:
                        self._dropped += excess
                        del self._tail[:excess]

    def text(self) -> str:
        with self._lock:
            head, tail, dropped = bytes(self._head), bytes(self._tail), self._dropped
        if not dropped:
            # One piece, so that a character across the two halves stays whole
            return (head + tail).decode(errors="replace")
        # Cut where characters begin, rather than through one
        kept_head = _complete_utf8(head)
        # Only what a cut character leaves, at most 3 bytes: output that is
        # not UTF-8 keeps its end, as replacement characters
        start = 0
        while start < min(3, len(tail)) and 0x80 <= tail[start] < 0xC0:
            start += 1
        kept_tail = tail[start:]
        dropped += len(head) - len(kept_head) + len(tail) - len(kept_tail)
        return (
            kept_head.decode(errors="replace")
            + f"\n[... {dropped} bytes of output truncated ...]\n"
            + kept_tail.decode(errors="replace")
        )
