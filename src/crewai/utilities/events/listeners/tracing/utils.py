import os
import platform
import uuid
import hashlib
import subprocess
import getpass
from pathlib import Path
from datetime import datetime
import re

import click

from crewai.utilities.paths import db_storage_path


def is_tracing_enabled() -> bool:
    return os.getenv("CREWAI_TRACING_ENABLED", "false").lower() == "true"


def on_first_execution_tracing_confirmation() -> bool:
    if is_first_execution():
        mark_first_execution_done()
        return click.confirm(
            "This is the first execution of CrewAI. Do you want to enable tracing?",
            default=False,
            show_default=True,
        )
    return False


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
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            if len(lines) >= 2:
                parts.append(lines[1])
    except Exception:
        pass

    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _user_id_file() -> Path:
    base = Path(db_storage_path()) / "state"
    base.mkdir(parents=True, exist_ok=True)
    return base / "user_id.txt"


def _load_cached_user_id() -> str | None:
    p = _user_id_file()
    if p.exists():
        try:
            val = p.read_text().strip()
            if val:
                return val
        except Exception:
            pass
    return None


def _cache_user_id(uid: str) -> None:
    try:
        p = _user_id_file()
        if not p.exists():
            p.write_text(uid + "\n")
    except Exception:
        pass


def get_user_id() -> str:
    """Stable, anonymized user identifier with caching and env override."""

    cached = _load_cached_user_id()
    if cached:
        return cached

    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    seed = f"{username}|{_get_machine_id()}"
    uid = hashlib.sha256(seed.encode()).hexdigest()
    _cache_user_id(uid)
    return uid


def _first_kickoff_marker_path() -> Path:
    base = Path(db_storage_path()) / "state"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"first_kickoff_{get_user_id()}.marker"


def is_first_execution() -> bool:
    return not _first_kickoff_marker_path().exists()


def mark_first_execution_done() -> None:
    marker = _first_kickoff_marker_path()
    if marker.exists():
        return
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(marker), flags, 0o644)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"created_at={datetime.now().timestamp()}\n")
            f.write(f"user_id={get_user_id()}\n")
            f.write(f"machine_id={_get_machine_id()}\n")
    except Exception:
        pass
