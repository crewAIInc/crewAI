"""Read and safely delegate work through the TaskMarket marketplace.

The tool intentionally separates public discovery from state-changing task creation.
Task creation is disabled unless the caller supplies the exact confirmation phrase
and a maximum spend that covers the requested reward. When authorized, creation is
handed to the first-party TaskMarket CLI; this tool never asks for, stores, or logs
wallet keys, seed phrases, passwords, or payment credentials.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
import json
import re
import shutil
import subprocess
from typing import Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


TASKMARKET_API_BASE = "https://api.taskmarket.dev/api"
TASKMARKET_TASK_URL_BASE = "https://taskmarket.dev/tasks"
CREATE_CONFIRMATION_PHRASE = "CREATE_TASKMARKET_TASK"
MICRO_USDC = Decimal("1000000")
SUPPORTED_MODES = {"bounty", "claim", "pitch", "benchmark", "auction"}
SUPPORTED_TASK_VISIBILITIES = {"public", "unlisted"}
SUPPORTED_SUBMISSION_VISIBILITIES = {
    "public",
    "reveal_all",
    "winner_only",
    "never",
}


class TaskMarketToolInput(BaseModel):
    """Input schema for :class:`TaskMarketTool`.

    ``create_task`` is intentionally opt-in and requires both
    ``confirmation`` and ``max_spend_usdc``. The creation workflow uses the
    caller's already configured first-party TaskMarket CLI context.
    """

    action: Literal[
        "list_tasks",
        "get_task",
        "list_submissions",
        "draft_task",
        "create_task",
    ] = Field(
        ...,
        description=(
            "Read-only actions are list_tasks, get_task, and list_submissions. "
            "draft_task creates no external state. create_task invokes the "
            "first-party TaskMarket CLI only after the exact confirmation phrase."
        ),
    )
    task_id: str | None = Field(
        default=None,
        description="TaskMarket task identifier. Required for get_task and list_submissions.",
    )
    status: str | None = Field(
        default="open",
        description="Optional TaskMarket status filter used only by list_tasks.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of tasks returned by list_tasks, from 1 to 100.",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Task tags for a draft or create request; up to 10 short tags.",
    )
    description: str | None = Field(
        default=None,
        description="Task description required for draft_task and create_task.",
    )
    deliverables: str | None = Field(
        default=None,
        description=(
            "Concrete deliverables required for draft_task and create_task. These are "
            "included in the TaskMarket description so workers can evaluate completion."
        ),
    )
    reward_usdc: str | None = Field(
        default=None,
        description="Requested reward in USDC, for example '2.50'. Required for drafts and creates.",
    )
    duration_hours: float | None = Field(
        default=None,
        gt=0,
        description="Task duration in hours. Required for drafts and creates.",
    )
    mode: Literal["bounty", "claim", "pitch", "benchmark", "auction"] = Field(
        default="bounty",
        description="TaskMarket task mode used for a draft or create request.",
    )
    task_visibility: Literal["public", "unlisted"] = Field(
        default="public",
        description="Public or unlisted task visibility. Private/password-protected tasks are intentionally unsupported.",
    )
    submission_visibility: Literal["public", "reveal_all", "winner_only", "never"] = (
        Field(
            default="public",
            description="How TaskMarket should expose submitted work for a draft or create request.",
        )
    )
    max_spend_usdc: str | None = Field(
        default=None,
        description="Maximum user-authorized USDC spend. Required and checked for create_task.",
    )
    confirmation: str | None = Field(
        default=None,
        description=(
            "For create_task only, enter the exact phrase "
            "CREATE_TASKMARKET_TASK after reviewing the draft."
        ),
    )


class TaskMarketTool(BaseTool):
    """Discover TaskMarket work and safely prepare an authorized task request.

    ``list_tasks``, ``get_task``, and ``list_submissions`` only use TaskMarket's
    public read API. ``draft_task`` returns the exact first-party CLI command
    preview and performs no network write. ``create_task`` is the only write path:
    it requires exact confirmation and invokes an already installed TaskMarket CLI
    without receiving any wallet secret or payment credential from this tool.

    The tool never claims tasks, submits work, accepts or rejects submissions,
    rates work, cancels work, or retries an uncertain creation request.
    """

    name: str = "TaskMarket Tool"
    description: str = (
        "Discover public TaskMarket tasks, inspect task details and submissions, "
        "draft a requester task, or—only after explicit confirmation—delegate an "
        "authorized task creation to the first-party TaskMarket CLI. This tool never "
        "claims, submits, accepts, rejects, rates, cancels, or retries marketplace work."
    )
    args_schema: type[BaseModel] = TaskMarketToolInput
    api_base_url: str = TASKMARKET_API_BASE
    task_url_base: str = TASKMARKET_TASK_URL_BASE
    cli_command: Literal["taskmarket"] = "taskmarket"
    request_timeout_seconds: int = 20
    cli_timeout_seconds: int = 120

    def _run(
        self,
        action: Literal[
            "list_tasks",
            "get_task",
            "list_submissions",
            "draft_task",
            "create_task",
        ],
        task_id: str | None = None,
        status: str | None = "open",
        limit: int = 20,
        tags: list[str] | None = None,
        description: str | None = None,
        deliverables: str | None = None,
        reward_usdc: str | None = None,
        duration_hours: float | None = None,
        mode: str = "bounty",
        task_visibility: str = "public",
        submission_visibility: str = "public",
        max_spend_usdc: str | None = None,
        confirmation: str | None = None,
    ) -> str:
        """Run a requested TaskMarket action with a strict write boundary."""
        tags = tags or []
        if action == "list_tasks":
            return self._list_tasks(status=status, limit=limit, tags=tags)
        if action == "get_task":
            return self._get_task(task_id)
        if action == "list_submissions":
            return self._list_submissions(task_id)

        payload_or_error = self._build_creation_payload(
            description=description,
            deliverables=deliverables,
            reward_usdc=reward_usdc,
            duration_hours=duration_hours,
            tags=tags,
            mode=mode,
            task_visibility=task_visibility,
            submission_visibility=submission_visibility,
        )
        if isinstance(payload_or_error, str):
            return payload_or_error
        payload = payload_or_error

        if action == "draft_task":
            return self._format_draft(payload)
        return self._create_task(
            payload=payload,
            max_spend_usdc=max_spend_usdc,
            confirmation=confirmation,
        )

    def _list_tasks(self, status: str | None, limit: int, tags: list[str]) -> str:
        params: dict[str, str | int] = {"limit": limit}
        if status:
            params["status"] = status
        if tags:
            params["tags"] = ",".join(tags)
        response_or_error = self._get("/tasks", params=params)
        if isinstance(response_or_error, str):
            return response_or_error
        data = response_or_error
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return "Error: TaskMarket returned an unexpected response shape."
        return json.dumps(
            {
                "task_count": len(tasks),
                "has_more": bool(data.get("hasMore", False)),
                "next_cursor": data.get("nextCursor"),
                "tasks": [self._task_summary(task) for task in tasks],
            },
            indent=2,
            sort_keys=True,
        )

    def _get_task(self, task_id: str | None) -> str:
        if not task_id:
            return "Error: task_id is required for get_task."
        data_or_error = self._get(f"/tasks/{task_id}")
        if isinstance(data_or_error, str):
            return data_or_error
        task = data_or_error.get("task", data_or_error)
        return json.dumps(self._task_summary(task, include_description=True), indent=2)

    def _list_submissions(self, task_id: str | None) -> str:
        if not task_id:
            return "Error: task_id is required for list_submissions."
        data_or_error = self._get(f"/tasks/{task_id}/submissions")
        if isinstance(data_or_error, str):
            return data_or_error
        return json.dumps(data_or_error, indent=2, sort_keys=True)

    def _get(
        self, path: str, params: dict[str, str | int] | None = None
    ) -> dict[str, object] | str:
        try:
            response = requests.get(
                f"{self.api_base_url}{path}",
                params=params,
                timeout=self.request_timeout_seconds,
            )
        except requests.RequestException as error:
            return f"Error: TaskMarket read request failed: {error!s}"
        if not response.ok:
            return f"Error: TaskMarket read request failed with HTTP {response.status_code}."
        try:
            data = response.json()
        except ValueError:
            return "Error: TaskMarket returned a non-JSON response."
        if not isinstance(data, dict):
            return "Error: TaskMarket returned an unexpected response shape."
        return data

    def _build_creation_payload(
        self,
        description: str | None,
        deliverables: str | None,
        reward_usdc: str | None,
        duration_hours: float | None,
        tags: list[str],
        mode: str,
        task_visibility: str,
        submission_visibility: str,
    ) -> dict[str, object] | str:
        if not description or not description.strip():
            return "Error: description is required for draft_task and create_task."
        if len(description) > 8_000:
            return "Error: description must not exceed 8,000 characters."
        if not deliverables or not deliverables.strip():
            return "Error: deliverables are required for draft_task and create_task."
        if len(deliverables) > 1_500:
            return "Error: deliverables must not exceed 1,500 characters."
        if duration_hours is None or duration_hours <= 0:
            return "Error: duration_hours must be a positive number."
        if not tags:
            return "Error: at least one tag is required for draft_task and create_task."
        if len(tags) > 10 or any(not tag.strip() for tag in tags):
            return "Error: provide between 1 and 10 non-empty tags."
        if (
            "\x00" in description
            or "\x00" in deliverables
            or any("\x00" in tag for tag in tags)
        ):
            return "Error: description, deliverables, and tags must not contain NUL characters."
        if mode not in SUPPORTED_MODES:
            return "Error: unsupported TaskMarket mode."
        if task_visibility not in SUPPORTED_TASK_VISIBILITIES:
            return (
                "Error: only public and unlisted visibility are supported by this tool."
            )
        if submission_visibility not in SUPPORTED_SUBMISSION_VISIBILITIES:
            return "Error: unsupported submission visibility."

        reward_or_error = self._parse_usdc(reward_usdc, "reward_usdc")
        if isinstance(reward_or_error, str):
            return reward_or_error
        reward = reward_or_error
        return {
            "description": f"{description.strip()}\n\n## Deliverables\n{deliverables.strip()}",
            "deliverables": deliverables.strip(),
            "reward_usdc": self._format_usdc(reward),
            "duration_hours": duration_hours,
            "tags": [tag.strip() for tag in tags],
            "mode": mode,
            "task_visibility": task_visibility,
            "submission_visibility": submission_visibility,
        }

    def _format_draft(self, payload: dict[str, object]) -> str:
        command = self._cli_command(payload)
        deadline_utc = (
            datetime.now(timezone.utc)
            + timedelta(hours=float(payload["duration_hours"]))
        ).isoformat()
        return json.dumps(
            {
                "network": "Base",
                "deadline_utc_estimate": deadline_utc,
                "write_performed": False,
                "exact_confirmation_required": CREATE_CONFIRMATION_PHRASE,
                "maximum_spend_required": payload["reward_usdc"],
                "task": payload,
                "first_party_cli_preview": command,
                "safety": (
                    "Review the task description, reward, duration, tags, and visibility "
                    "before creating it. This draft cannot claim, submit, accept, reject, "
                    "rate, cancel, or otherwise alter marketplace work."
                ),
            },
            indent=2,
            sort_keys=True,
        )

    def _create_task(
        self,
        payload: dict[str, object],
        max_spend_usdc: str | None,
        confirmation: str | None,
    ) -> str:
        if confirmation != CREATE_CONFIRMATION_PHRASE:
            return (
                "Creation was not attempted. To create a TaskMarket task, review the "
                f"draft and provide confirmation='{CREATE_CONFIRMATION_PHRASE}'."
            )
        max_spend_or_error = self._parse_usdc(max_spend_usdc, "max_spend_usdc")
        if isinstance(max_spend_or_error, str):
            return max_spend_or_error
        reward_or_error = self._parse_usdc(str(payload["reward_usdc"]), "reward_usdc")
        if isinstance(reward_or_error, str):
            return reward_or_error
        if max_spend_or_error < reward_or_error:
            return (
                "Creation was not attempted because max_spend_usdc is lower than the "
                "requested reward."
            )
        if shutil.which(self.cli_command) is None:
            return (
                "Creation was not attempted because the first-party TaskMarket CLI was "
                f"not found: {self.cli_command}. Install and configure it outside this tool, "
                "then review the draft again."
            )

        command = self._cli_command(payload)
        try:
            result = subprocess.run(  # noqa: S603  # Command is a fixed first-party CLI with validated list arguments.
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=self.cli_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return (
                "Task creation timed out. Settlement status is unknown, so this tool did "
                "not retry. Check TaskMarket directly before taking any further action."
            )
        except OSError as error:
            return f"Task creation was not attempted because the CLI could not start: {error!s}"

        output = (result.stdout or result.stderr or "").strip()
        if result.returncode != 0:
            return (
                "TaskMarket CLI reported an unsuccessful creation attempt. This tool did "
                "not retry. Review the CLI output and TaskMarket directly before trying again. "
                f"CLI output: {output[:2000]}"
            )

        task_id = self._extract_task_id(output)
        response: dict[str, object] = {
            "created": True,
            "network": "Base",
            "cli_output": output[:2000],
            "not_retried": True,
        }
        if task_id:
            response["task_id"] = task_id
            response["task_url"] = f"{self.task_url_base}/{task_id}"
        else:
            response["status_check_required"] = (
                "The CLI did not return a recognizable task ID. Check TaskMarket directly; "
                "this tool will not retry an uncertain creation."
            )
        return json.dumps(response, indent=2, sort_keys=True)

    def _cli_command(self, payload: dict[str, object]) -> list[str]:
        return [
            self.cli_command,
            "task",
            "create",
            "--description",
            str(payload["description"]),
            "--reward",
            str(payload["reward_usdc"]),
            "--duration",
            str(payload["duration_hours"]),
            "--mode",
            str(payload["mode"]),
            "--task-visibility",
            str(payload["task_visibility"]),
            "--submission-visibility",
            str(payload["submission_visibility"]),
            "--tags",
            ",".join(str(tag) for tag in payload["tags"]),
        ]

    @staticmethod
    def _parse_usdc(value: str | None, field_name: str) -> Decimal | str:
        if value is None:
            return f"Error: {field_name} is required."
        try:
            amount = Decimal(value)
        except (InvalidOperation, ValueError):
            return f"Error: {field_name} must be a positive USDC decimal amount."
        if not amount.is_finite() or amount <= 0:
            return f"Error: {field_name} must be a positive USDC decimal amount."
        if amount.as_tuple().exponent < -6:
            return f"Error: {field_name} supports at most 6 decimal places."
        return amount

    @staticmethod
    def _format_usdc(amount: Decimal) -> str:
        return (
            format(amount.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP), "f")
            .rstrip("0")
            .rstrip(".")
        )

    @staticmethod
    def _extract_task_id(output: str) -> str | None:
        try:
            decoded = json.loads(output)
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            for key in ("taskId", "task_id", "id"):
                value = decoded.get(key)
                if isinstance(value, str) and value:
                    return value
        match = re.search(r"0x[a-fA-F0-9]{64}", output)
        return match.group(0) if match else None

    @staticmethod
    def _task_summary(
        task: object, include_description: bool = False
    ) -> dict[str, object]:
        if not isinstance(task, dict):
            return {"raw": task}
        summary = {
            "id": task.get("id"),
            "status": task.get("status"),
            "mode": task.get("mode"),
            "reward_micro_usdc": task.get("reward"),
            "reward_usdc": TaskMarketTool._micro_usdc_to_usdc(task.get("reward")),
            "net_reward_micro_usdc": task.get("netReward"),
            "net_reward_usdc": TaskMarketTool._micro_usdc_to_usdc(
                task.get("netReward")
            ),
            "expiry_time": task.get("expiryTime"),
            "tags": task.get("tags", []),
            "submission_count": task.get("submissionCount"),
            "task_url": (
                f"{TASKMARKET_TASK_URL_BASE}/{task['id']}"
                if isinstance(task.get("id"), str)
                else None
            ),
        }
        if include_description:
            summary["description"] = task.get("description")
        return summary

    @staticmethod
    def _micro_usdc_to_usdc(value: object) -> str | None:
        if not isinstance(value, (str, int, float)):
            return None
        try:
            micro_usdc = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        return TaskMarketTool._format_usdc(micro_usdc / MICRO_USDC)
