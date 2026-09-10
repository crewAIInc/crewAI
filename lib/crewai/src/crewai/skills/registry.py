"""Registry reference resolution for the Agent Skills standard.

Handles @org/skill-name references (optionally pinned to a version with
@org/skill-name@version), local-first resolution, and downloads via the CrewAI
AMP API with a global cache at ~/.crewai/skills/.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

from crewai.skills.cache import SkillCacheManager
from crewai.skills.validation import versions_match


_logger = logging.getLogger(__name__)


class SkillRef(NamedTuple):
    """A parsed registry reference.

    Attributes:
        org: Organisation slug or uuid.
        name: Skill name.
        version: Pinned version, or None to resolve the latest.
    """

    org: str
    name: str
    version: str | None = None

    def __str__(self) -> str:
        ref = f"@{self.org}/{self.name}"
        return f"{ref}@{self.version}" if self.version else ref


def is_registry_ref(value: Any) -> bool:
    """Return True if *value* looks like a registry reference (@org/name)."""
    return isinstance(value, str) and value.startswith("@")


def parse_skill_ref(ref: str) -> SkillRef:
    """Parse a registry reference into its organisation, name and version.

    Accepts '@org/skill-name' and the version-pinned '@org/skill-name@1.2.0'
    (a leading 'v' on the version is allowed: '@org/skill-name@v1.2.0').

    Args:
        ref: A registry reference.

    Returns:
        The parsed :class:`SkillRef`.

    Raises:
        ValueError: If the reference format is invalid.
    """
    if not ref.startswith("@"):
        raise ValueError(f"Registry reference must start with '@', got: {ref!r}")
    without_at = ref[1:]
    if without_at.count("/") != 1:
        raise ValueError(
            f"Registry reference must be in '@org/name' format, got: {ref!r}"
        )
    org, remainder = without_at.split("/", 1)
    # Skill names cannot contain '@', so the first one starts the version pin.
    name, separator, version = remainder.partition("@")
    version = version.strip()
    if (
        not org
        or not name
        or org.startswith(".")
        or name.startswith(".")
        or "/" in org
        or "/" in name
    ):
        raise ValueError(
            f"Registry reference org and name must be single, non-empty path "
            f"segments (no '..' or leading dots), got: {ref!r}"
        )
    if separator and not version:
        raise ValueError(
            f"Registry reference version must be non-empty when pinned, got: {ref!r}"
        )
    if "@" in version:
        raise ValueError(
            f"Registry reference must carry a single version pin, got: {ref!r}"
        )
    return SkillRef(org=org, name=name, version=version or None)


def parse_registry_ref(ref: str) -> tuple[str, str]:
    """Parse '@org/skill-name' into (org, name), ignoring any version pin.

    Retained for backwards compatibility. Use :func:`parse_skill_ref` to read
    the pinned version too.

    Args:
        ref: A registry reference, e.g. '@acme/my-skill'.

    Returns:
        An (org, name) tuple.

    Raises:
        ValueError: If the reference format is invalid.
    """
    parsed = parse_skill_ref(ref)
    return parsed.org, parsed.name


def resolve_registry_ref(
    ref: str,
    source: Any = None,
    *,
    activate: bool = True,
) -> Skill:  # type: ignore[name-defined]  # noqa: F821
    """Resolve a registry reference to a Skill object.

    Resolution order:
    1. ./skills/{name}/ in the current working directory (project-local)
    2. ~/.crewai/skills/{org}/{name}/ (global cache)
    3. Download from registry

    A version-pinned reference only accepts a local or cached copy that reports
    the pinned version, and downloads otherwise — a pin is a request for a
    specific version, not a hint.

    Args:
        ref: A registry reference, e.g. '@acme/my-skill' or '@acme/my-skill@1.2.0'.
        source: Optional source object passed through to skill loaders (for events).
        activate: Whether to load the full SKILL.md instructions.

    Returns:
        A Skill loaded at INSTRUCTIONS level by default, or METADATA level
        when ``activate`` is false.
    """
    from crewai.skills.loader import activate_skill

    org, name, version = parse_skill_ref(ref)

    local_path = Path.cwd() / "skills" / name
    if local_path.is_dir() and (local_path / "SKILL.md").exists():
        # A project-local copy has no installation metadata, so its frontmatter
        # is the only thing a pin can be checked against.
        skill = _load_matching_skill(local_path, version)
        if skill is not None:
            return activate_skill(skill, source=source) if activate else skill

    cache = SkillCacheManager()
    cached_path = cache.get_cached_path(org, name, version=version)
    if cached_path is not None and (cached_path / "SKILL.md").exists():
        # get_cached_path already matched the pin against the version the cache
        # recorded when it stored the archive, which is authoritative. Checking
        # the frontmatter as well would miss on every skill that doesn't declare
        # metadata.version and re-download it on each resolution.
        skill = _load_skill(cached_path)
        if skill is not None:
            return activate_skill(skill, source=source) if activate else skill

    return download_skill(
        org,
        name,
        source=source,
        version=version,
        activate=activate,
    )


def _load_skill(path: Path) -> Skill | None:  # type: ignore[name-defined]  # noqa: F821
    """Load the skill at *path*, or None when it can't be read."""
    from crewai.skills.parser import load_skill_metadata

    try:
        return load_skill_metadata(path)
    except Exception:
        _logger.debug("Failed to load skill at %s", path, exc_info=True)
        return None


def _load_matching_skill(
    path: Path,
    version: str | None,
) -> Skill | None:  # type: ignore[name-defined]  # noqa: F821
    """Load the skill at *path*, or None when it can't satisfy *version*.

    Skills record their version in SKILL.md frontmatter under
    ``metadata.version``; a skill that declares no version can't satisfy a pin.
    """
    skill = _load_skill(path)
    if skill is None or version is None:
        return skill

    declared = (skill.frontmatter.metadata or {}).get("version")
    if versions_match(version, declared):
        return skill

    _logger.debug(
        "Skill at %s declares version %r, which does not satisfy the pinned %r",
        path,
        declared,
        version,
    )
    return None


def _accepts_version(get_skill: Any) -> bool:
    """Return True when *get_skill* takes a ``version`` argument.

    Assumes it does when the callable can't be introspected, so an unusual but
    working client isn't pushed onto the fallback path.
    """
    try:
        parameters = inspect.signature(get_skill).parameters
    except (TypeError, ValueError):
        return True

    return "version" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _plus_client(ref: str, pinned: bool) -> Any:
    """Return the CrewAI AMP client to fetch skills with.

    Hosted runtimes install a client authenticated for the environment they run
    in, and have no user credential to fall back on — so skills resolve through
    the same client the Agent Repository uses rather than building a ``PlusAPI``
    of their own.

    Args:
        ref: The reference being resolved, for logging.
        pinned: Whether this lookup carries a version pin, which the installed
            client has to be able to forward.
    """
    from crewai.utilities.agent_utils import resolve_plus_client

    def build_default_client() -> Any:
        # Deliberately wider than the Agent Repository's default, which reads the
        # CLI login only: skills kept their existing precedence so this fix
        # doesn't change which credential either lookup picks.
        from crewai_core.plus_api import PlusAPI

        from crewai.auth.token import get_auth_token
        from crewai.context import get_platform_integration_token

        return PlusAPI(
            api_key=os.getenv("CREWAI_USER_PAT")
            or get_platform_integration_token()
            or get_auth_token(),
            organization_id=os.getenv("CREWAI_ORGANIZATION_UUID"),
        )

    client = resolve_plus_client(build_default_client)
    get_skill = getattr(client, "get_skill", None)

    # A runtime predating the Skills Repository installs a client that can't
    # fetch skills at all; one predating pins has a get_skill that would reject
    # the version argument. Repository agents pin automatically, so both cases
    # fall back to a client that can serve the request rather than raising.
    if not callable(get_skill) or (pinned and not _accepts_version(get_skill)):
        _logger.warning(
            "The configured CrewAI AMP client cannot fetch %s, falling back to "
            "credentials from the environment.",
            "pinned skills" if pinned else "skills",
        )
        return build_default_client()

    return client


def download_skill(
    org: str,
    name: str,
    source: Any = None,
    *,
    version: str | None = None,
    activate: bool = True,
) -> Skill:  # type: ignore[name-defined]  # noqa: F821
    """Download a skill from the registry and store it in the cache.

    Args:
        org: Organisation slug.
        name: Skill name.
        source: Optional source for event emission.
        version: Optional pinned version; the latest is fetched when omitted.
        activate: Whether to load the full SKILL.md instructions.

    Returns:
        The downloaded Skill at INSTRUCTIONS level by default, or METADATA
        level when ``activate`` is false.

    Raises:
        ValueError: If *version* is given but blank.
    """
    from crewai.skills.loader import activate_skill
    from crewai.skills.parser import load_skill_metadata
    from crewai.utilities.agent_utils import resolve_plus_response

    if version is not None:
        # A blank pin would otherwise read as "unpinned" and quietly float to
        # the latest version, which is not what a caller passing one asked for.
        version = version.strip()
        if not version:
            raise ValueError(
                "A pinned skill version must be non-empty; omit it to fetch the latest."
            )

    ref = str(SkillRef(org=org, name=name, version=version))

    try:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.skills.events import (
            SkillDownloadCompletedEvent,
            SkillDownloadStartedEvent,
        )

        _has_events = True
    except ImportError:
        _has_events = False

    if _has_events:
        crewai_event_bus.emit(
            source,
            event=SkillDownloadStartedEvent(
                registry_ref=ref,
            ),
        )

    try:
        api = _plus_client(ref, pinned=bool(version))
        # Only pass the version when pinned, so clients predating version
        # support keep working for unpinned refs. The pin goes over verbatim and
        # the registry decides how to match it, including a leading "v".
        pin = {"version": version} if version else {}
        response = resolve_plus_response(api.get_skill(org, name, **pin))
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download skill {ref!r} from registry: {exc}"
        ) from exc

    import base64

    import httpx

    served_version = data.get("version") or data.get("latest_version")
    if version and not versions_match(version, served_version):
        # Registries that predate pinning ignore the parameter and serve their
        # newest version. Caching that under the pinned version would poison
        # every later lookup for the pin, so refuse it instead.
        raise RuntimeError(
            f"Failed to download skill {ref!r} from registry: it served version "
            f"{served_version!r} rather than the pinned version {version!r}."
        )

    resolved_version = version or served_version

    download_url = data.get("download_url")
    if download_url:
        dl_response = httpx.get(download_url, follow_redirects=True)
        dl_response.raise_for_status()
        archive_bytes = dl_response.content
    else:
        encoded = data.get("file", "")
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        archive_bytes = base64.b64decode(encoded)

    cache = SkillCacheManager()
    skill_dir = cache.store(org, name, resolved_version, archive_bytes)

    if _has_events:
        crewai_event_bus.emit(
            source,
            event=SkillDownloadCompletedEvent(
                registry_ref=ref,
                version=resolved_version,
                cache_path=skill_dir,
            ),
        )

    if not (skill_dir / "SKILL.md").exists():
        raise RuntimeError(
            f"Skill archive for {ref!r} downloaded but no SKILL.md found in {skill_dir}"
        )
    skill = load_skill_metadata(skill_dir)
    return activate_skill(skill, source=source) if activate else skill
