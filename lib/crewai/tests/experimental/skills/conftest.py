import pytest


@pytest.fixture(autouse=True)
def _enable_experimental_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_EXPERIMENTAL", "1")