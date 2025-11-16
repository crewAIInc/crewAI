from contextvars import ContextVar, Token
from datetime import datetime
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, cast
import uuid

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from crewai.utilities.paths import db_storage_path
from crewai.utilities.serialization import to_serializable


logger = logging.getLogger(__name__)


_tracing_enabled: ContextVar[bool | None] = ContextVar("_tracing_enabled", default=None)


def should_enable_tracing(*, override: bool | None = None) -> bool:
    """Determine if tracing should be enabled.

    This is the single source of truth for tracing enablement.
    Priority order:
    1. Explicit override (e.g., Crew.tracing=True/False)
    2. Environment variable CREWAI_TRACING_ENABLED
    3. User consent from user_data

    Args:
        override: Explicit override for tracing (True=always enable, False=always disable, None=check other settings)

    Returns:
        True if tracing should be enabled, False otherwise.
    """
    if override is True:
        return True
    if override is False:
        return False

    env_value = os.getenv("CREWAI_TRACING_ENABLED", "").lower()
    if env_value in ("true", "1"):
        return True

    data = _load_user_data()

    if data.get("trace_consent", False) is not False:
        return True

    return False


def set_tracing_enabled(enabled: bool) -> object:
    """Set tracing enabled state for current execution context.

    Args:
        enabled: Whether tracing should be enabled

    Returns:
        A token that can be used with reset_tracing_enabled to restore previous value.
    """
    return _tracing_enabled.set(enabled)


def reset_tracing_enabled(token: Token[bool | None]) -> None:
    """Reset tracing enabled state to previous value.

    Args:
        token: Token returned from set_tracing_enabled
    """
    _tracing_enabled.reset(token)


def is_tracing_enabled_in_context() -> bool:
    """Check if tracing is enabled in current execution context.

    Returns:
        True if tracing is enabled in context, False otherwise.
        Returns False if context has not been set.
    """
    enabled = _tracing_enabled.get()
    return enabled if enabled is not None else False


def _user_data_file() -> Path:
    base = Path(db_storage_path())
    base.mkdir(parents=True, exist_ok=True)
    return base / ".crewai_user.json"


def _load_user_data() -> dict[str, Any]:
    p = _user_data_file()
    if p.exists():
        try:
            return cast(dict[str, Any], json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, PermissionError) as e:
            logger.warning(f"Failed to load user data: {e}")
    return {}


def _save_user_data(data: dict[str, Any]) -> None:
    try:
        p = _user_data_file()
        p.write_text(json.dumps(data, indent=2))
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to save user data: {e}")


def has_user_declined_tracing() -> bool:
    """Check if user has explicitly declined trace collection.

    Returns:
        True if user previously declined tracing, False otherwise.
    """
    data = _load_user_data()
    if data.get("first_execution_done", False):
        return data.get("trace_consent", False) is False
    return False


def is_tracing_enabled() -> bool:
    """Check if tracing should be enabled.


    Returns:
        True if tracing is enabled and not disabled, False otherwise.
    """
    # If user has explicitly declined tracing, never enable it
    if has_user_declined_tracing():
        return False

    return os.getenv("CREWAI_TRACING_ENABLED", "false").lower() == "true"


def on_first_execution_tracing_confirmation() -> bool:
    if _is_test_environment():
        return False

    if is_first_execution():
        mark_first_execution_done()
        return click.confirm(
            "This is the first execution of CrewAI. Do you want to enable tracing?",
            default=True,
            show_default=True,
        )
    return False


def _is_test_environment() -> bool:
    """Detect if we're running in a test environment."""
    return os.environ.get("CREWAI_TESTING", "").lower() == "true"


def _get_machine_id() -> str:
    """Stable, privacy-preserving machine fingerprint (cross-platform)."""
    parts = []

    try:
        mac = ":".join(
            [f"{(uuid.getnode() >> b) & 0xFF:02x}" for b in range(0, 12, 2)][::-1]
        )
        parts.append(mac)
    except Exception:  # noqa: S110
        pass

    try:
        sysname = platform.system()
        parts.append(sysname)
    except Exception:
        sysname = "unknown"
        parts.append(sysname)

    try:
        if sysname == "Darwin":
            try:
                res = subprocess.run(
                    ["/usr/sbin/system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                m = re.search(r"Hardware UUID:\s*([A-Fa-f0-9\-]+)", res.stdout)
                if m:
                    parts.append(m.group(1))
            except Exception:  # noqa: S110
                pass

        elif sysname == "Linux":
            linux_id = _get_linux_machine_id()
            if linux_id:
                parts.append(linux_id)

        elif sysname == "Windows":
            try:
                res = subprocess.run(
                    [
                        "C:\\Windows\\System32\\wbem\\wmic.exe",
                        "csproduct",
                        "get",
                        "UUID",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                lines = [
                    line.strip() for line in res.stdout.splitlines() if line.strip()
                ]
                if len(lines) >= 2:
                    parts.append(lines[1])
            except Exception:  # noqa: S110
                pass
        else:
            generic_id = _get_generic_system_id()
            if generic_id:
                parts.append(generic_id)

    except Exception:  # noqa: S110
        pass

    if len(parts) <= 1:
        try:
            import socket

            parts.append(socket.gethostname())
        except Exception:  # noqa: S110
            pass

        try:
            parts.append(getpass.getuser())
        except Exception:  # noqa: S110
            pass

        try:
            parts.append(platform.machine())
            parts.append(platform.processor())
        except Exception:  # noqa: S110
            pass

    if not parts:
        parts.append("unknown-system")
        parts.append(str(uuid.uuid4()))

    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _get_linux_machine_id() -> str | None:
    linux_id_sources = [
        "/etc/machine-id",
        "/sys/class/dmi/id/product_uuid",
        "/proc/sys/kernel/random/boot_id",
        "/sys/class/dmi/id/board_serial",
        "/sys/class/dmi/id/chassis_serial",
    ]

    for source in linux_id_sources:
        try:
            path = Path(source)
            if path.exists() and path.is_file():
                content = path.read_text().strip()
                if content and content.lower() not in [
                    "unknown",
                    "to be filled by o.e.m.",
                    "",
                ]:
                    return content
        except Exception:  # noqa: S112, PERF203
            continue

    try:
        import socket

        hostname = socket.gethostname()
        arch = platform.machine()
        if hostname and arch:
            return f"{hostname}-{arch}"
    except Exception:  # noqa: S110
        pass

    return None


def _get_generic_system_id() -> str | None:
    try:
        parts = []

        try:
            import socket

            hostname = socket.gethostname()
            if hostname:
                parts.append(hostname)
        except Exception:  # noqa: S110
            pass

        try:
            parts.append(platform.machine())
            parts.append(platform.processor())
            parts.append(platform.architecture()[0])
        except Exception:  # noqa: S110
            pass

        try:
            container_id = os.environ.get(
                "HOSTNAME", os.environ.get("CONTAINER_ID", "")
            )
            if container_id:
                parts.append(container_id)
        except Exception:  # noqa: S110
            pass

        if parts:
            return "-".join(filter(None, parts))

    except Exception:  # noqa: S110
        pass

    return None


def get_user_id() -> str:
    """Stable, anonymized user identifier with caching."""
    data = _load_user_data()

    if "user_id" in data:
        return cast(str, data["user_id"])

    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    seed = f"{username}|{_get_machine_id()}"
    uid = hashlib.sha256(seed.encode()).hexdigest()

    data["user_id"] = uid
    _save_user_data(data)
    return uid


def is_first_execution() -> bool:
    """True if this is the first execution for this user."""
    data = _load_user_data()
    return not data.get("first_execution_done", False)


def mark_first_execution_done(user_consented: bool = False) -> None:
    """Mark that the first execution has been completed.

    Args:
        user_consented: Whether the user consented to trace collection.
    """
    data = _load_user_data()
    if data.get("first_execution_done", False):
        return

    data.update(
        {
            "first_execution_done": True,
            "first_execution_at": datetime.now().timestamp(),
            "user_id": get_user_id(),
            "machine_id": _get_machine_id(),
            "trace_consent": user_consented,
        }
    )
    _save_user_data(data)


def safe_serialize_to_dict(obj: Any, exclude: set[str] | None = None) -> dict[str, Any]:
    """Safely serialize an object to a dictionary for event data."""
    try:
        serialized = to_serializable(obj, exclude)
        if isinstance(serialized, dict):
            return serialized
        return {"serialized_data": serialized}
    except Exception as e:
        return {"serialization_error": str(e), "object_type": type(obj).__name__}


def truncate_messages(
    messages: list[dict[str, Any]], max_content_length: int = 500, max_messages: int = 5
) -> list[dict[str, Any]]:
    """Truncate message content and limit number of messages"""
    if not messages or not isinstance(messages, list):
        return messages

    limited_messages = messages[:max_messages]

    for msg in limited_messages:
        if isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
            if len(content) > max_content_length:
                msg["content"] = content[:max_content_length] + "..."

    return limited_messages


def should_auto_collect_first_time_traces() -> bool:
    """True if we should auto-collect traces for first-time user.


    Returns:
        True if first-time user AND telemetry not disabled AND tracing not explicitly enabled, False otherwise.
    """
    if _is_test_environment():
        return False

    # If user has previously declined, never auto-collect
    if has_user_declined_tracing():
        return False

    if is_tracing_enabled_in_context():
        return False

    return is_first_execution()


def prompt_user_for_trace_viewing(timeout_seconds: int = 20) -> bool:
    """
    Prompt user if they want to see their traces with timeout.
    Returns True if user wants to see traces, False otherwise.
    """
    if _is_test_environment():
        return False

    try:
        import threading

        console = Console()

        content = Text()
        content.append("🔍 ", style="cyan bold")
        content.append(
            "Detailed execution traces are available!\n\n", style="cyan bold"
        )
        content.append("View insights including:\n", style="white")
        content.append("  • Agent decision-making process\n", style="bright_blue")
        content.append("  • Task execution flow and timing\n", style="bright_blue")
        content.append("  • Tool usage details", style="bright_blue")

        panel = Panel(
            content,
            title="[bold cyan]Execution Traces[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
        console.print("\n")
        console.print(panel)

        prompt_text = click.style(
            f"Would you like to view your execution traces? [y/N] ({timeout_seconds}s timeout): ",
            fg="white",
            bold=True,
        )
        click.echo(prompt_text, nl=False)

        result = [False]

        def get_input() -> None:
            try:
                response = input().strip().lower()
                result[0] = response in ["y", "yes"]
            except (EOFError, KeyboardInterrupt, OSError, LookupError):
                # Handle all input-related errors silently
                result[0] = False

        input_thread = threading.Thread(target=get_input, daemon=True)
        input_thread.start()
        input_thread.join(timeout=timeout_seconds)

        if input_thread.is_alive():
            return False

        return result[0]

    except Exception:
        # Suppress any warnings or errors and assume "no"
        return False


def mark_first_execution_completed(user_consented: bool = False) -> None:
    """Mark first execution as completed (called after trace prompt).

    Args:
        user_consented: Whether the user consented to trace collection.
    """
    mark_first_execution_done(user_consented=user_consented)
