import os
import platform
import uuid
import hashlib
import subprocess
import getpass
from pathlib import Path
from datetime import datetime
import re
import json

import click

from crewai.utilities.paths import db_storage_path


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
            ["{:02x}".format((uuid.getnode() >> b) & 0xFF) for b in range(0, 12, 2)][
                ::-1
            ]
        )
        parts.append(mac)
    except Exception:
        pass

    sysname = platform.system()
    parts.append(sysname)

    try:
        if sysname == "Darwin":
            res = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
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
                ["wmic", "csproduct", "get", "UUID"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if len(lines) >= 2:
                parts.append(lines[1])
    except Exception:
        pass

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
        except Exception:
            pass
    return {}


def _save_user_data(data: dict) -> None:
    try:
        p = _user_data_file()
        p.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


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
