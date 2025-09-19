import getpass
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from crewai.utilities.paths import db_storage_path
from crewai.utilities.serialization import to_serializable

logger = logging.getLogger(__name__)


def is_tracing_enabled() -> bool:
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
    except Exception:
        logger.warning("Error getting machine id for fingerprinting")

    sysname = platform.system()
    parts.append(sysname)

    try:
        if sysname == "Darwin":
            res = subprocess.run(
                ["/usr/sbin/system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            m = re.search(r"Hardware UUID:\s*([A-Fa-f0-9\-]+)", res.stdout)
            if m:
                parts.append(m.group(1))
        elif sysname == "Linux":
            try:
                parts.append(Path("/etc/machine-id").read_text().strip())
            except Exception:
                parts.append(Path("/sys/class/dmi/id/product_uuid").read_text().strip())
        elif sysname == "Windows":
            res = subprocess.run(
                ["C:\\Windows\\System32\\wbem\\wmic.exe", "csproduct", "get", "UUID"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if len(lines) >= 2:
                parts.append(lines[1])
    except Exception:
        logger.exception("Error getting machine ID")

    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _user_data_file() -> Path:
    base = Path(db_storage_path())
    base.mkdir(parents=True, exist_ok=True)
    return base / ".crewai_user.json"


def _load_user_data() -> dict:
    p = _user_data_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError, PermissionError) as e:
            logger.warning(f"Failed to load user data: {e}")
    return {}


def _save_user_data(data: dict) -> None:
    try:
        p = _user_data_file()
        p.write_text(json.dumps(data, indent=2))
    except (OSError, PermissionError) as e:
        logger.warning(f"Failed to save user data: {e}")


def get_user_id() -> str:
    """Stable, anonymized user identifier with caching."""
    data = _load_user_data()

    if "user_id" in data:
        return data["user_id"]

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


def mark_first_execution_done() -> None:
    """Mark that the first execution has been completed."""
    data = _load_user_data()
    if data.get("first_execution_done", False):
        return

    data.update(
        {
            "first_execution_done": True,
            "first_execution_at": datetime.now().timestamp(),
            "user_id": get_user_id(),
            "machine_id": _get_machine_id(),
        }
    )
    _save_user_data(data)


def safe_serialize_to_dict(obj, exclude: set[str] | None = None) -> dict[str, Any]:
    """Safely serialize an object to a dictionary for event data."""
    try:
        serialized = to_serializable(obj, exclude)
        if isinstance(serialized, dict):
            return serialized
        return {"serialized_data": serialized}
    except Exception as e:
        return {"serialization_error": str(e), "object_type": type(obj).__name__}


def truncate_messages(messages, max_content_length=500, max_messages=5):
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
    """True if we should auto-collect traces for first-time user."""
    if _is_test_environment():
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

        def get_input():
            try:
                response = input().strip().lower()
                result[0] = response in ["y", "yes"]
            except (EOFError, KeyboardInterrupt):
                result[0] = False

        input_thread = threading.Thread(target=get_input, daemon=True)
        input_thread.start()
        input_thread.join(timeout=timeout_seconds)

        if input_thread.is_alive():
            return False

        return result[0]

    except Exception:
        return False


def mark_first_execution_completed() -> None:
    """Mark first execution as completed (called after trace prompt)."""
    mark_first_execution_done()
