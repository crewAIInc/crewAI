"""On-disk storage for traces and skill proposals.

Layout::

    <root>/
        traces/<role>/<run_id>.json
        skill_proposals/<role>/<proposal_id>.json
        skills/<role>/<skill-name>/SKILL.md

The default root is ``db_storage_path() / "self_improve"`` — the same
project-scoped, platform-correct data dir that memory DBs use (set by
``appdirs.user_data_dir`` and overridable via ``CREWAI_STORAGE_DIR``).
``CREWAI_SELF_IMPROVE_DIR`` overrides specifically this feature's root,
useful for tests or migrations.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

from crewai.skills.self_improve.models import RunTrace, SkillProposal
from crewai.utilities.paths import db_storage_path


_ENV_VAR = "CREWAI_SELF_IMPROVE_DIR"
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def _slug(role: str) -> str:
    """Slugify an agent role for use as a directory name."""
    s = role.strip().lower().replace(" ", "-")
    s = _SLUG_RE.sub("", s)
    return s or "agent"


def _resolve_root(root: Path | None) -> Path:
    if root is not None:
        return root
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    return Path(db_storage_path()) / "self_improve"


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


class TraceStore:
    """Filesystem store for ``RunTrace`` records."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = _resolve_root(root) / "traces"

    def role_dir(self, role: str) -> Path:
        return self.root / _slug(role)

    def path_for(self, trace: RunTrace) -> Path:
        return self.role_dir(trace.agent_role) / f"{trace.id}.json"

    def save(self, trace: RunTrace) -> Path:
        path = self.path_for(trace)
        _write_json(path, trace.model_dump_json(indent=2))
        return path

    def list_for_role(self, role: str) -> list[Path]:
        d = self.role_dir(role)
        if not d.exists():
            return []
        return sorted(d.glob("*.json"))

    def load(self, path: Path) -> RunTrace:
        return RunTrace.model_validate_json(path.read_text(encoding="utf-8"))

    def count_for_role(self, role: str) -> int:
        return len(self.list_for_role(role))


class ProposalStore:
    """Filesystem store for ``SkillProposal`` records pending human review."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = _resolve_root(root) / "skill_proposals"

    def role_dir(self, role: str) -> Path:
        return self.root / _slug(role)

    def path_for(self, proposal: SkillProposal) -> Path:
        return self.role_dir(proposal.agent_role) / f"{proposal.id}.json"

    def save(self, proposal: SkillProposal) -> Path:
        path = self.path_for(proposal)
        _write_json(path, proposal.model_dump_json(indent=2))
        return path

    def list_for_role(self, role: str) -> list[Path]:
        d = self.role_dir(role)
        if not d.exists():
            return []
        return sorted(d.glob("*.json"))

    def load(self, path: Path) -> SkillProposal:
        return SkillProposal.model_validate_json(path.read_text(encoding="utf-8"))

    def delete(self, proposal_id: str, role: str) -> bool:
        path = self.role_dir(role) / f"{proposal_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def find(self, proposal_id: str) -> SkillProposal | None:
        """Locate a proposal by id across all role dirs. Returns None if missing."""
        if not self.root.exists():
            return None
        for role_dir in self.root.iterdir():
            if not role_dir.is_dir():
                continue
            path = role_dir / f"{proposal_id}.json"
            if path.exists():
                return self.load(path)
        return None

    def list_all(self) -> list[SkillProposal]:
        """All proposals across roles, oldest first by file id."""
        if not self.root.exists():
            return []
        out: list[SkillProposal] = []
        for role_dir in sorted(self.root.iterdir()):
            if not role_dir.is_dir():
                continue
            for path in sorted(role_dir.glob("*.json")):
                out.append(self.load(path))
        return out


class SkillStore:
    """Filesystem store for accepted (live) skills.

    Each accepted ``SkillProposal`` becomes a directory under
    ``role_dir(role) / skill_name`` with a ``SKILL.md`` inside, matching
    the layout the existing skill loader discovers.

    Two ways to construct:

    - ``SkillStore()`` — root is ``<db_storage_path>/self_improve/skills/``
      (the platform default colocated with traces + proposals).
    - ``SkillStore(skills_root=Path("./skills/learned"))`` — root is the
      given path verbatim. Use this when the agent is configured with
      ``SelfImprovementConfig(skills_dir=...)`` so accepted skills land in
      the project tree.
    """

    def __init__(
        self,
        root: Path | None = None,
        skills_root: Path | None = None,
    ) -> None:
        self.root = (
            skills_root
            if skills_root is not None
            else _resolve_root(root) / "skills"
        )

    def role_dir(self, role: str) -> Path:
        return self.root / _slug(role)

    def skill_dir(self, role: str, skill_name: str) -> Path:
        return self.role_dir(role) / skill_name

    def has_any(self, role: str) -> bool:
        d = self.role_dir(role)
        if not d.exists():
            return False
        return any((child / "SKILL.md").is_file() for child in d.iterdir() if child.is_dir())
