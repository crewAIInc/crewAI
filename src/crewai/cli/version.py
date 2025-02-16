import importlib.metadata


def get_crewai_version() -> str:
    """Get the version number of CrewAI running the CLI"""
    return importlib.metadata.version("crewai")
