"""Guarded requester workflow for Taskmarket."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import shutil

# Required for the fixed first-party CLI and always invoked without a shell.
import subprocess  # nosec B404
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


BASE_CHAIN_ID = 8453
BASE_USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
TASK_ID_PATTERN = re.compile(r"^0x[0-9a-fA-F]{64}$")

Operation = Literal["prepare_create", "create", "status", "submissions"]
TaskVisibility = Literal["public", "unlisted"]
SubmissionVisibility = Literal["public", "reveal_all", "winner_only", "never"]
CommandRunner = Callable[[Sequence[str], float], tuple[int, str, str]]


class TaskMarketRequesterSchema(BaseModel):
    """Inputs exposed to a CrewAI agent."""

    operation: Operation = Field(
        ...,
        description=(
            "Operation to perform. prepare_create only prepares an exact preview; "
            "create requires separate host approval of that preview."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Exact bounty brief, required only for prepare_create.",
    )
    deliverables: list[str] | None = Field(
        default=None,
        description="Concrete deliverables, required only for prepare_create.",
    )
    reward_usdc: Decimal | None = Field(
        default=None,
        gt=0,
        description="Gross bounty reward in human-readable USDC.",
    )
    duration_hours: int | None = Field(
        default=None,
        ge=1,
        le=24 * 30,
        description="Submission window in whole hours.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Optional Taskmarket discovery tags.",
    )
    task_visibility: TaskVisibility = Field(
        default="public",
        description="Public or unlisted task visibility. Onchain activity stays public.",
    )
    submission_visibility: SubmissionVisibility = Field(
        default="public",
        description="Permanent submission visibility selected at task creation.",
    )
    preview_id: str | None = Field(
        default=None,
        description="Approved preview identifier, required only for create.",
    )
    task_id: str | None = Field(
        default=None,
        description="0x-prefixed 32-byte task ID for status or submissions.",
    )


@dataclass(frozen=True)
class _PreparedCreate:
    preview_id: str
    fingerprint: str
    description: str
    deliverables: tuple[str, ...]
    reward_usdc: Decimal
    duration_hours: int
    tags: tuple[str, ...]
    task_visibility: TaskVisibility
    submission_visibility: SubmissionVisibility
    prepared_at: datetime
    projected_deadline: datetime

    @property
    def cli_args(self) -> list[str]:
        """Return the exact argument vector approved for task creation."""
        args = [
            "task",
            "create",
            "--description",
            self.description,
            "--reward",
            _format_usdc(self.reward_usdc),
            "--duration",
            str(self.duration_hours),
            "--mode",
            "bounty",
            "--task-visibility",
            self.task_visibility,
            "--submission-visibility",
            self.submission_visibility,
        ]
        if self.tags:
            args.extend(["--tags", ",".join(self.tags)])
        return args


class _UnknownSettlementError(RuntimeError):
    """The CLI attempt ended without a trustworthy settlement result."""


def _format_usdc(value: Decimal) -> str:
    """Format a validated USDC amount with the canonical six decimals."""
    return format(value.quantize(Decimal("0.000001")), "f")


def _json_result(payload: dict[str, Any]) -> str:
    """Serialize a stable, human-readable tool result."""
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


class TaskMarketRequesterTool(BaseTool):
    """Create and monitor Taskmarket bounties with host-controlled approval.

    The model can prepare a byte-for-byte preview, but cannot approve it. Trusted
    application code must call :meth:`approve` with the displayed preview ID and
    fingerprint before the model-facing ``create`` operation can spend USDC.
    """

    name: str = "Taskmarket requester"
    description: str = (
        "Prepare an exact Taskmarket bounty for human approval, execute only a "
        "host-approved preview, retrieve live task status, or present submissions "
        "for human review. Never accepts or rejects submissions."
    )
    args_schema: type[BaseModel] = TaskMarketRequesterSchema

    max_reward_usdc: Decimal = Field(
        default=Decimal("10"),
        gt=0,
        description="Hard cap for one bounty reward and maximum USDC spend.",
    )
    cli_path: str = Field(
        default="taskmarket",
        description="First-party Taskmarket CLI executable.",
    )
    command_timeout_seconds: float = Field(
        default=45,
        gt=0,
        description="Timeout for one CLI call. Timed-out writes are never retried.",
    )
    approval_ttl_seconds: int = Field(
        default=300,
        ge=1,
        le=900,
        description="Seconds before explicit host approval expires.",
    )

    _runner: CommandRunner = PrivateAttr()
    _previews: dict[str, _PreparedCreate] = PrivateAttr(default_factory=dict)
    _approved: dict[str, datetime] = PrivateAttr(default_factory=dict)
    _attempted: set[str] = PrivateAttr(default_factory=set)
    _lock: Any = PrivateAttr(default_factory=Lock)

    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the requester tool.

        Args:
            runner: Optional argument-array runner for tests or host isolation.
            **kwargs: BaseTool and policy configuration.
        """
        super().__init__(**kwargs)
        self._runner = runner or self._default_runner

    def approve(self, preview_id: str, fingerprint: str) -> None:
        """Approve one exact prepared action from trusted host code.

        Args:
            preview_id: Identifier returned by ``prepare_create``.
            fingerprint: SHA-256 fingerprint shown with that preview.

        Raises:
            ValueError: If the preview is missing, changed, or already attempted.
        """
        with self._lock:
            preview = self._previews.get(preview_id)
            if preview is None or preview.fingerprint != fingerprint:
                raise ValueError("Preview ID or fingerprint does not match.")
            if preview_id in self._attempted:
                raise ValueError(
                    "This preview was already attempted and cannot be reused."
                )
            self._approved[preview_id] = datetime.now(UTC)

    def _run(
        self,
        operation: Operation,
        description: str | None = None,
        deliverables: list[str] | None = None,
        reward_usdc: Decimal | None = None,
        duration_hours: int | None = None,
        tags: list[str] | None = None,
        task_visibility: TaskVisibility = "public",
        submission_visibility: SubmissionVisibility = "public",
        preview_id: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """Run the selected requester operation."""
        if operation == "prepare_create":
            return self._prepare_create(
                description=description,
                deliverables=deliverables,
                reward_usdc=reward_usdc,
                duration_hours=duration_hours,
                tags=tags,
                task_visibility=task_visibility,
                submission_visibility=submission_visibility,
            )
        if operation == "create":
            return self._create(preview_id)
        if operation == "status":
            return self._read_task(task_id)
        if operation == "submissions":
            return self._read_submissions(task_id)
        raise ValueError(f"Unsupported operation: {operation}")

    def _prepare_create(
        self,
        *,
        description: str | None,
        deliverables: list[str] | None,
        reward_usdc: Decimal | None,
        duration_hours: int | None,
        tags: list[str] | None,
        task_visibility: TaskVisibility,
        submission_visibility: SubmissionVisibility,
    ) -> str:
        """Validate inputs and prepare a non-spending, approval-bound preview."""
        if not description or not description.strip():
            raise ValueError("description is required for prepare_create.")
        if not deliverables or not all(item.strip() for item in deliverables):
            raise ValueError("At least one non-empty deliverable is required.")
        if reward_usdc is None or duration_hours is None:
            raise ValueError("reward_usdc and duration_hours are required.")
        if len(description) > 8_000:
            raise ValueError("description must not exceed 8,000 characters.")
        if len(deliverables) > 30 or any(len(item) > 500 for item in deliverables):
            raise ValueError("Provide at most 30 deliverables of 500 characters each.")
        if duration_hours < 1 or duration_hours > 24 * 30:
            raise ValueError("duration_hours must be between 1 and 720.")
        if "\x00" in description or any("\x00" in item for item in deliverables):
            raise ValueError("Description and deliverables cannot contain NUL bytes.")
        try:
            supplied_reward = Decimal(reward_usdc)
        except InvalidOperation as exc:
            raise ValueError("reward_usdc must be a decimal amount.") from exc
        if not supplied_reward.is_finite():
            raise ValueError("reward_usdc must be finite.")
        try:
            reward = supplied_reward.quantize(Decimal("0.000001"))
        except InvalidOperation as exc:
            raise ValueError("reward_usdc must have at most six decimals.") from exc
        if reward != supplied_reward:
            raise ValueError("reward_usdc must have at most six decimals.")
        if reward <= 0 or reward > self.max_reward_usdc:
            raise ValueError(
                f"reward_usdc must be above zero and at most {self.max_reward_usdc}."
            )

        clean_tags = tuple(
            dict.fromkeys(tag.strip() for tag in tags or [] if tag.strip())
        )
        if len(clean_tags) > 10 or any(len(tag) > 64 for tag in clean_tags):
            raise ValueError("Provide at most 10 tags of 64 characters each.")
        if any("," in tag for tag in clean_tags):
            raise ValueError("Tags cannot contain commas.")
        if any("\x00" in tag for tag in clean_tags):
            raise ValueError("Tags cannot contain NUL bytes.")
        exact_description = (
            description.strip()
            + "\n\nDeliverables:\n"
            + "\n".join(f"- {item.strip()}" for item in deliverables)
        )
        prepared_at = datetime.now(UTC)
        projected_deadline = prepared_at + timedelta(hours=duration_hours)
        canonical = {
            "description": exact_description,
            "duration_hours": duration_hours,
            "mode": "bounty",
            "reward_usdc": _format_usdc(reward),
            "submission_visibility": submission_visibility,
            "tags": clean_tags,
            "task_visibility": task_visibility,
        }
        fingerprint = hashlib.sha256(
            json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        preview_id = uuid4().hex
        preview = _PreparedCreate(
            preview_id=preview_id,
            fingerprint=fingerprint,
            description=exact_description,
            deliverables=tuple(item.strip() for item in deliverables),
            reward_usdc=reward,
            duration_hours=duration_hours,
            tags=clean_tags,
            task_visibility=task_visibility,
            submission_visibility=submission_visibility,
            prepared_at=prepared_at,
            projected_deadline=projected_deadline,
        )
        self._previews[preview_id] = preview
        return _json_result(
            {
                "approval_expires_seconds": self.approval_ttl_seconds,
                "approval_required": True,
                "base_chain_id": BASE_CHAIN_ID,
                "description": exact_description,
                "deliverables": preview.deliverables,
                "deadline_policy": f"{duration_hours} hours after onchain creation",
                "duration_hours": duration_hours,
                "fingerprint_sha256": fingerprint,
                "maximum_spend_usdc": _format_usdc(reward),
                "network": "Base mainnet",
                "preview_id": preview_id,
                "projected_deadline_utc": projected_deadline.isoformat(),
                "reward_usdc": _format_usdc(reward),
                "submission_visibility": submission_visibility,
                "task_visibility": task_visibility,
                "trusted_host_next_step": (
                    "Display this exact preview, then call tool.approve(preview_id, "
                    "fingerprint_sha256) only after fresh user authorization."
                ),
            }
        )

    def _create(self, preview_id: str | None) -> str:
        """Consume one fresh host approval and attempt task creation once."""
        with self._lock:
            if not preview_id or preview_id not in self._previews:
                raise ValueError("A valid preview_id is required for create.")
            if preview_id in self._attempted:
                raise PermissionError(
                    "This preview was already attempted and cannot be retried."
                )
            approved_at = self._approved.get(preview_id)
            if approved_at is None:
                raise PermissionError(
                    "This exact preview has not been approved by trusted host code."
                )
            if datetime.now(UTC) - approved_at > timedelta(
                seconds=self.approval_ttl_seconds
            ):
                self._approved.pop(preview_id, None)
                raise PermissionError(
                    "Approval expired; display the exact preview and request fresh "
                    "authorization before creating."
                )
            preview = self._previews[preview_id]
            self._approved.pop(preview_id, None)
            self._attempted.add(preview_id)

        preflight = self._preflight(preview.reward_usdc)
        try:
            created = self._cli(preview.cli_args, write=True)
        except _UnknownSettlementError as exc:
            return _json_result(
                {
                    "created": "unknown",
                    "error": str(exc),
                    "preview_id": preview_id,
                    "retry_allowed": False,
                    "required_action": (
                        "Reconcile with Taskmarket inbox and wallet history. Do not retry "
                        "this preview because settlement may have succeeded."
                    ),
                }
            )

        data = created.get("data", {})
        task_id = data.get("taskId") or data.get("id")
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            return _json_result(
                {
                    "created": "unknown",
                    "preview_id": preview_id,
                    "raw_result": created,
                    "retry_allowed": False,
                    "required_action": "Reconcile with Taskmarket inbox before any new write.",
                }
            )

        try:
            live_status: dict[str, Any] | None = self._cli(["task", "get", task_id])
        except RuntimeError:
            live_status = None
        return _json_result(
            {
                "created": True,
                "maximum_spend_usdc": _format_usdc(preview.reward_usdc),
                "network": preflight["network"],
                "preview_id": preview_id,
                "retry_allowed": False,
                "task_id": task_id,
                "task_url": f"https://taskmarket.dev/tasks/{task_id}",
                "wallet_address": preflight["wallet_address"],
                "live_status": live_status,
            }
        )

    def _preflight(self, reward: Decimal) -> dict[str, str]:
        """Fail closed unless network, legal state, and balance are safe."""
        address_data = self._cli(["address"]).get("data", {})
        deposit_data = self._cli(["deposit"]).get("data", {})
        balance_data = self._cli(["wallet", "balance"]).get("data", {})
        legal_data = self._cli(["legal", "status"]).get("data", {})

        if deposit_data.get("chainId") != BASE_CHAIN_ID:
            raise RuntimeError("Taskmarket CLI is not configured for Base mainnet.")
        contract = str(deposit_data.get("usdcContract", "")).lower()
        if contract != BASE_USDC:
            raise RuntimeError(
                "Taskmarket CLI returned a non-canonical Base USDC contract."
            )
        enforcement_enabled = legal_data.get("enforcementEnabled")
        if not isinstance(enforcement_enabled, bool):
            raise RuntimeError(
                "Taskmarket CLI did not report the legal enforcement state."
            )
        if enforcement_enabled and legal_data.get("accepted") is not True:
            raise RuntimeError(
                "The current Taskmarket legal bundle requires acceptance."
            )
        try:
            balance = Decimal(str(balance_data["balanceUsdc"]))
        except (InvalidOperation, KeyError) as exc:
            raise RuntimeError(
                "Taskmarket CLI returned an invalid USDC balance."
            ) from exc
        if not balance.is_finite():
            raise RuntimeError(
                "Taskmarket CLI returned a non-finite USDC balance."
            )
        if balance < reward:
            raise RuntimeError(
                f"Insufficient Base USDC: need {_format_usdc(reward)}, have {balance}."
            )
        wallet_address = address_data.get("address")
        if not isinstance(wallet_address, str) or not wallet_address.strip():
            raise RuntimeError(
                "Taskmarket CLI did not return the acting wallet address."
            )
        return {"network": "Base mainnet", "wallet_address": wallet_address}

    def _read_task(self, task_id: str | None) -> str:
        """Retrieve live task state without enabling a marketplace write."""
        exact_id = self._validate_task_id(task_id)
        return _json_result(
            {
                "human_review_required": True,
                "task": self._cli(["task", "get", exact_id]),
                "task_url": f"https://taskmarket.dev/tasks/{exact_id}",
            }
        )

    def _read_submissions(self, task_id: str | None) -> str:
        """Retrieve submissions for human review without judging them."""
        exact_id = self._validate_task_id(task_id)
        return _json_result(
            {
                "accept_or_reject_enabled": False,
                "human_review_required": True,
                "submissions": self._cli(["task", "submissions", exact_id]),
                "task_id": exact_id,
            }
        )

    @staticmethod
    def _validate_task_id(task_id: str | None) -> str:
        """Return a structurally valid Taskmarket task identifier."""
        if not task_id or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("task_id must be a 0x-prefixed 32-byte hexadecimal value.")
        return task_id

    def _cli(self, args: list[str], *, write: bool = False) -> dict[str, Any]:
        """Run the fixed first-party CLI and validate its JSON envelope."""
        try:
            returncode, stdout, stderr = self._runner(
                [self.cli_path, *args], self.command_timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            if write:
                raise _UnknownSettlementError(
                    "The create command timed out without a trustworthy settlement result."
                ) from exc
            raise RuntimeError("Taskmarket CLI read timed out.") from exc

        result = self._parse_envelope(stdout, stderr)
        if returncode != 0 or result.get("ok") is not True:
            message = result.get("error", "Taskmarket CLI command failed.")
            if write:
                raise _UnknownSettlementError(str(message))
            raise RuntimeError(str(message))
        if not isinstance(result.get("data", {}), dict):
            message = "Taskmarket CLI returned a malformed data payload."
            if write:
                raise _UnknownSettlementError(message)
            raise RuntimeError(message)
        return result

    @staticmethod
    def _parse_envelope(stdout: str, stderr: str) -> dict[str, Any]:
        """Find the last Taskmarket JSON envelope in either output stream."""
        for stream in (stdout, stderr):
            for line in reversed(stream.splitlines()):
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "ok" in parsed:
                    return parsed
        return {"ok": False, "error": "Taskmarket CLI returned no JSON envelope."}

    @staticmethod
    def _default_runner(args: Sequence[str], timeout: float) -> tuple[int, str, str]:
        """Execute an argument vector without a shell and capture its output."""
        executable = shutil.which(args[0])
        if executable is None:
            raise RuntimeError("First-party taskmarket CLI is not installed.")
        # Arguments remain an array and the executable is resolved explicitly; no shell.
        completed = subprocess.run(  # noqa: S603  # nosec B603
            [executable, *args[1:]],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout, completed.stderr
