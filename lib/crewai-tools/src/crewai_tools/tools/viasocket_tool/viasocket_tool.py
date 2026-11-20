import os

from crewai.tools import BaseTool


def ViaSocketTools(  # noqa: N802
    user: str,
    apps: list[str],
    actions: dict[str, list[str]] | None = None,
    org_id: str | None = None,
    project_id: str | None = None,
    secret: str | None = None,
) -> list[BaseTool]:
    """Factory function that returns viaSocket action tools for one end user.

    Each chosen action of an app the user connected through viaSocket (Slack,
    Google Sheets, HubSpot and 2,300+ more) becomes one tool, run on that
    user's own connection.

    Args:
        user: The end user's unique identifier their apps were connected under.
        apps: The apps, by slug (``slack``, ``google-sheets``).
        actions: Optional actions per app, by ``action_id`` or path
            (``{"slack": ["send-message"]}``). An app left out gets every
            published action.
        org_id: viaSocket org id. Defaults to ``VIASOCKET_ORG_ID``.
        project_id: viaSocket embed project id. Defaults to ``VIASOCKET_PROJECT_ID``.
        secret: viaSocket embed secret. Defaults to ``VIASOCKET_EMBED_SECRET``.

    Returns:
        A list of viaSocket action tools.
    """
    try:
        from viasocket import ViaSocket
        from viasocket.crewai import viasocket_tools
    except ImportError as error:
        raise ImportError(
            "The 'viasocket' package is required for ViaSocketTools. "
            "Install it with: uv add viasocket"
        ) from error

    values = {
        "VIASOCKET_ORG_ID": org_id or os.getenv("VIASOCKET_ORG_ID"),
        "VIASOCKET_PROJECT_ID": project_id or os.getenv("VIASOCKET_PROJECT_ID"),
        "VIASOCKET_EMBED_SECRET": secret or os.getenv("VIASOCKET_EMBED_SECRET"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"{', '.join(missing)} is not set")

    viasocket = ViaSocket(
        values["VIASOCKET_ORG_ID"],
        values["VIASOCKET_PROJECT_ID"],
        values["VIASOCKET_EMBED_SECRET"],
    )
    return list(viasocket_tools(viasocket, user=user, apps=apps, actions=actions))
