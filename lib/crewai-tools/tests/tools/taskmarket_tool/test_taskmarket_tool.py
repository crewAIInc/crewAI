import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from crewai_tools.tools import TaskMarketTool
from crewai_tools.tools.taskmarket_tool.taskmarket_tool import (
    CREATE_CONFIRMATION_PHRASE,
)


def _create_arguments() -> dict[str, object]:
    return {
        "description": "Review a public documentation set and return cited findings.",
        "deliverables": "One Markdown report with source URLs and a concise findings table.",
        "reward_usdc": "2.50",
        "duration_hours": 24,
        "tags": ["research", "docs"],
        "mode": "bounty",
        "task_visibility": "public",
        "submission_visibility": "winner_only",
    }


def test_list_tasks_formats_public_task_summaries(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.ok = True
    response.json.return_value = {
        "tasks": [
            {
                "id": "0x" + "a" * 64,
                "status": "open",
                "mode": "bounty",
                "reward": "2500000",
                "netReward": "2312500",
                "expiryTime": "2026-08-20T00:00:00.000Z",
                "tags": ["research"],
                "submissionCount": 1,
            }
        ],
        "hasMore": False,
        "nextCursor": None,
    }
    get = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "get", get)

    result = json.loads(TaskMarketTool()._run(action="list_tasks", limit=5))

    get.assert_called_once_with(
        "https://api.taskmarket.dev/api/tasks",
        params={"limit": 5, "status": "open"},
        timeout=20,
    )
    assert result["task_count"] == 1
    assert result["tasks"][0]["reward_usdc"] == "2.5"
    assert result["tasks"][0]["net_reward_usdc"] == "2.3125"
    assert result["tasks"][0]["task_url"].endswith("a" * 64)


def test_get_task_includes_description(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.ok = True
    response.json.return_value = {
        "task": {
            "id": "0x" + "b" * 64,
            "status": "open",
            "reward": "1000000",
            "description": "A complete public task description.",
        }
    }
    monkeypatch.setattr(requests, "get", MagicMock(return_value=response))

    result = json.loads(
        TaskMarketTool()._run(action="get_task", task_id="0x" + "b" * 64)
    )

    assert result["description"] == "A complete public task description."
    assert result["reward_usdc"] == "1"


def test_list_tasks_rejects_non_list_tasks_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    response.ok = True
    response.json.return_value = {"tasks": None}
    monkeypatch.setattr(requests, "get", MagicMock(return_value=response))

    result = TaskMarketTool()._run(action="list_tasks")

    assert result == "Error: TaskMarket returned an unexpected response shape."


def test_list_tasks_returns_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        MagicMock(side_effect=requests.RequestException("offline")),
    )

    result = TaskMarketTool()._run(action="list_tasks")

    assert result == "Error: TaskMarket read request failed: offline"


def test_draft_never_runs_cli() -> None:
    tool = TaskMarketTool()

    result = json.loads(tool._run(action="draft_task", **_create_arguments()))

    assert result["write_performed"] is False
    assert result["exact_confirmation_required"] == CREATE_CONFIRMATION_PHRASE
    assert result["maximum_spend_required"] == "2.5"
    assert result["task"]["duration_hours"] == 24
    assert result["task"]["deliverables"].startswith("One Markdown report")
    assert "## Deliverables" in result["task"]["description"]
    assert result["deadline_utc_estimate"].endswith("+00:00")
    assert result["first_party_cli_preview"][:3] == ["taskmarket", "task", "create"]


def test_draft_requires_explicit_deliverables() -> None:
    arguments = _create_arguments()
    arguments.pop("deliverables")

    result = TaskMarketTool()._run(action="draft_task", **arguments)

    assert result == "Error: deliverables are required for draft_task and create_task."


def test_create_requires_exact_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("shutil.which", MagicMock(return_value="/usr/local/bin/taskmarket"))

    result = TaskMarketTool()._run(
        action="create_task",
        max_spend_usdc="2.50",
        confirmation="yes",
        **_create_arguments(),
    )

    assert "Creation was not attempted" in result
    run.assert_not_called()


def test_create_rejects_insufficient_maximum_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("shutil.which", MagicMock(return_value="/usr/local/bin/taskmarket"))

    result = TaskMarketTool()._run(
        action="create_task",
        max_spend_usdc="2.49",
        confirmation=CREATE_CONFIRMATION_PHRASE,
        **_create_arguments(),
    )

    assert "max_spend_usdc is lower" in result
    run.assert_not_called()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("description", "invalid\x00description"),
        ("deliverables", "invalid\x00deliverables"),
        ("tags", ["research", "invalid\x00tag"]),
    ],
)
def test_create_rejects_nul_without_invoking_cli(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid_value: str | list[str],
) -> None:
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("shutil.which", MagicMock(return_value="/usr/local/bin/taskmarket"))
    arguments = _create_arguments()
    arguments[field_name] = invalid_value

    result = TaskMarketTool()._run(
        action="create_task",
        max_spend_usdc="3.00",
        confirmation=CREATE_CONFIRMATION_PHRASE,
        **arguments,
    )

    assert "must not contain NUL characters" in result
    run.assert_not_called()


def test_create_uses_first_party_cli_after_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = MagicMock(
        return_value=SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"taskId": "0x" + "c" * 64}),
            stderr="",
        )
    )
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("shutil.which", MagicMock(return_value="/usr/local/bin/taskmarket"))

    result = json.loads(
        TaskMarketTool()._run(
            action="create_task",
            max_spend_usdc="3.00",
            confirmation=CREATE_CONFIRMATION_PHRASE,
            **_create_arguments(),
        )
    )

    command = run.call_args.args[0]
    assert command[:3] == ["taskmarket", "task", "create"]
    assert "--reward" in command
    assert command[command.index("--reward") + 1] == "2.5"
    assert result["created"] is True
    assert result["task_id"] == "0x" + "c" * 64
    assert result["task_url"].endswith("c" * 64)


def test_create_timeout_never_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    run = MagicMock(side_effect=subprocess.TimeoutExpired("taskmarket", 120))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("shutil.which", MagicMock(return_value="/usr/local/bin/taskmarket"))

    result = TaskMarketTool()._run(
        action="create_task",
        max_spend_usdc="3.00",
        confirmation=CREATE_CONFIRMATION_PHRASE,
        **_create_arguments(),
    )

    assert "Settlement status is unknown" in result
    assert "did not retry" in result
    assert run.call_count == 1
