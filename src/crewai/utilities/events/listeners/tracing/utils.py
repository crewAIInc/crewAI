import os


def is_tracing_enabled() -> bool:
    return os.getenv("CREWAI_TRACING_ENABLED", "false").lower() == "true"
