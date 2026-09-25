"""CrewAI Platform application catalog."""

from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Final, TypeAlias


PlatformApp: TypeAlias = str


@dataclass(frozen=True)
class PlatformToolDefinition:
    """Describe one selectable CrewAI Platform action."""

    slug: str
    display_name: str


def _load_platform_catalog() -> tuple[
    tuple[str, ...],
    dict[str, str],
    dict[str, int],
    dict[str, tuple[PlatformToolDefinition, ...]],
]:
    """Load and validate the generated Clipper application catalog."""
    catalog_path = files("crewai_core").joinpath("platform_catalog.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    applications = catalog.get("applications") if isinstance(catalog, dict) else None
    if not isinstance(applications, list):
        raise ValueError("Platform catalog must contain an 'applications' list.")

    apps: list[str] = []
    display_names: dict[str, str] = {}
    tool_counts: dict[str, int] = {}
    app_tools: dict[str, tuple[PlatformToolDefinition, ...]] = {}
    for application in applications:
        if not isinstance(application, dict):
            raise ValueError("Every platform catalog application must be an object.")
        slug = application.get("slug")
        display_name = application.get("display_name")
        if not isinstance(slug, str) or not slug:
            raise ValueError("Every platform catalog application must have a slug.")
        if not isinstance(display_name, str) or not display_name:
            raise ValueError(
                f"Platform catalog application '{slug}' must have a display name."
            )
        if slug in display_names:
            raise ValueError(f"Platform catalog contains duplicate slug '{slug}'.")
        tools = application.get("tools", [])
        if not isinstance(tools, list):
            raise ValueError(
                f"Platform catalog application '{slug}' must have a tools list."
            )
        parsed_tools: list[PlatformToolDefinition] = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError(
                    f"Platform catalog application '{slug}' contains an invalid tool."
                )
            tool_slug = tool.get("slug")
            tool_display_name = tool.get("display_name")
            if not isinstance(tool_slug, str) or not tool_slug:
                raise ValueError(
                    f"Platform catalog application '{slug}' contains a tool without a slug."
                )
            if not isinstance(tool_display_name, str) or not tool_display_name:
                raise ValueError(
                    f"Platform catalog tool '{slug}/{tool_slug}' must have a display name."
                )
            parsed_tools.append(
                PlatformToolDefinition(
                    slug=tool_slug,
                    display_name=tool_display_name,
                )
            )
        apps.append(slug)
        display_names[slug] = display_name
        tool_counts[slug] = len(parsed_tools)
        app_tools[slug] = tuple(parsed_tools)

    return tuple(apps), display_names, tool_counts, app_tools


(
    _platform_apps,
    _platform_app_display_names,
    _platform_app_tool_counts,
    _platform_app_tools,
) = _load_platform_catalog()
PLATFORM_APPS: Final[tuple[str, ...]] = _platform_apps
PLATFORM_APP_DISPLAY_NAMES: Final[dict[str, str]] = _platform_app_display_names
PLATFORM_APP_TOOL_COUNTS: Final[dict[str, int]] = _platform_app_tool_counts
PLATFORM_APP_TOOLS: Final[dict[str, tuple[PlatformToolDefinition, ...]]] = (
    _platform_app_tools
)
