from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal
import json
import subprocess
from threading import Event

from crewai_tools import TaskMarketRequesterTool
from crewai_tools.tools.taskmarket_requester_tool import TaskMarketRequesterSchema
from pydantic import ValidationError
import pytest


TASK_ID = "0x" + "a" * 64
WALLET = "0x1111111111111111111111111111111111111111"
BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


class RecordedRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.overrides: dict[tuple[str, ...], dict[str, object]] = {}
        self.create_exception: Exception | None = None
        self.address_started: Event | None = None
        self.release_address: Event | None = None
        self.create_result: dict[str, object] = {
            "ok": True,
            "data": {"taskId": TASK_ID},
        }

    def __call__(
        self, args: Sequence[str], timeout: float
    ) -> tuple[int, str, str]:
        assert timeout == 45
        command = tuple(args[1:])
        self.calls.append(command)
        if command == ("address",) and self.address_started is not None:
            self.address_started.set()
            assert self.release_address is not None
            assert self.release_address.wait(timeout=5)
        if command[:2] == ("task", "create"):
            if self.create_exception is not None:
                raise self.create_exception
            payload = self.create_result
        else:
            payload = self.overrides.get(command, self._default(command))
        return (0 if payload.get("ok") is True else 1, json.dumps(payload), "")

    @staticmethod
    def _default(command: tuple[str, ...]) -> dict[str, object]:
        responses: dict[tuple[str, ...], dict[str, object]] = {
            ("address",): {"ok": True, "data": {"address": WALLET}},
            ("deposit",): {
                "ok": True,
                "data": {"chainId": 8453, "usdcContract": BASE_USDC},
            },
            ("wallet", "balance"): {
                "ok": True,
                "data": {"balanceUsdc": "10.000000"},
            },
            ("legal", "status"): {
                "ok": True,
                "data": {"enforcementEnabled": False, "accepted": False},
            },
            ("task", "get", TASK_ID): {
                "ok": True,
                "data": {"id": TASK_ID, "status": "open"},
            },
            ("task", "submissions", TASK_ID): {
                "ok": True,
                "data": {"submissions": []},
            },
        }
        if command not in responses:
            raise AssertionError(f"Unexpected Taskmarket command: {command}")
        return responses[command]


@pytest.fixture
def runner() -> RecordedRunner:
    return RecordedRunner()


@pytest.fixture
def tool(runner: RecordedRunner) -> TaskMarketRequesterTool:
    return TaskMarketRequesterTool(
        runner=runner,
        max_reward_usdc=Decimal("5"),
    )


def prepare(tool: TaskMarketRequesterTool, **overrides: object) -> dict[str, object]:
    inputs: dict[str, object] = {
        "operation": "prepare_create",
        "description": "Audit the release candidate.",
        "deliverables": ["Written report", "Reproduction steps"],
        "reward_usdc": Decimal("1.25"),
        "duration_hours": 24,
        "tags": ["audit", "python"],
        "task_visibility": "public",
        "submission_visibility": "winner_only",
    }
    inputs.update(overrides)
    return json.loads(tool.run(**inputs))


def approve_prepared(tool: TaskMarketRequesterTool) -> dict[str, object]:
    preview = prepare(tool)
    tool.approve(
        str(preview["preview_id"]),
        str(preview["fingerprint_sha256"]),
    )
    return preview


def test_prepare_shows_exact_action_without_calling_cli(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    preview = prepare(tool)

    assert preview["approval_required"] is True
    assert preview["approval_expires_seconds"] == 300
    assert preview["description"] == (
        "Audit the release candidate.\n\nDeliverables:\n"
        "- Written report\n- Reproduction steps"
    )
    assert preview["deliverables"] == ["Written report", "Reproduction steps"]
    assert preview["reward_usdc"] == "1.250000"
    assert preview["maximum_spend_usdc"] == "1.250000"
    assert preview["duration_hours"] == 24
    assert preview["deadline_policy"] == "24 hours after onchain creation"
    assert preview["network"] == "Base mainnet"
    assert preview["base_chain_id"] == 8453
    assert preview["task_visibility"] == "public"
    assert preview["submission_visibility"] == "winner_only"
    assert str(preview["projected_deadline_utc"]).endswith("+00:00")
    assert len(str(preview["fingerprint_sha256"])) == 64
    assert runner.calls == []


def test_public_tool_entrypoint_validates_and_prepares(
    tool: TaskMarketRequesterTool,
) -> None:
    preview = json.loads(
        tool.run(
            operation="prepare_create",
            description="Audit the release candidate.",
            deliverables=["Written report"],
            reward_usdc=Decimal("1.25"),
            duration_hours=24,
        )
    )

    assert preview["approval_required"] is True
    assert preview["reward_usdc"] == "1.250000"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"description": "  "}, "description is required"),
        ({"deliverables": []}, "At least one non-empty deliverable"),
        ({"reward_usdc": Decimal("5.000001")}, "at most 5"),
        ({"reward_usdc": Decimal("1.0000001")}, "at most six decimals"),
        ({"description": "x" * 8_001}, "must not exceed 8,000"),
        ({"description": "unsafe\x00description"}, "cannot contain NUL"),
        ({"deliverables": ["x" * 501]}, "at most 30 deliverables"),
        ({"deliverables": ["unsafe\x00deliverable"]}, "cannot contain NUL"),
        ({"tags": ["safe,unsafe"]}, "Tags cannot contain commas"),
        ({"tags": [str(index) for index in range(11)]}, "at most 10 tags"),
        ({"tags": ["unsafe\x00tag"]}, "Tags cannot contain NUL"),
    ],
)
def test_prepare_rejects_unsafe_or_ambiguous_inputs(
    tool: TaskMarketRequesterTool,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare(tool, **overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reward_usdc": Decimal("Infinity")}, "must be finite"),
        ({"duration_hours": 0}, "between 1 and 720"),
        ({"duration_hours": 721}, "between 1 and 720"),
    ],
)
def test_direct_run_repeats_guards_enforced_by_schema(
    tool: TaskMarketRequesterTool,
    overrides: dict[str, object],
    message: str,
) -> None:
    inputs: dict[str, object] = {
        "operation": "prepare_create",
        "description": "Audit the release candidate.",
        "deliverables": ["Written report"],
        "reward_usdc": Decimal("1.25"),
        "duration_hours": 24,
    }
    inputs.update(overrides)

    with pytest.raises(ValueError, match=message):
        tool._run(**inputs)


def test_create_requires_fresh_exact_host_approval(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    preview = prepare(tool)
    preview_id = str(preview["preview_id"])

    with pytest.raises(PermissionError, match="not been approved"):
        tool._run(operation="create", preview_id=preview_id)
    with pytest.raises(ValueError, match="does not match"):
        tool.approve(preview_id, "0" * 64)

    assert runner.calls == []


def test_expired_approval_requires_fresh_authorization(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    preview = approve_prepared(tool)
    preview_id = str(preview["preview_id"])
    tool._approved[preview_id] -= timedelta(seconds=tool.approval_ttl_seconds + 1)

    with pytest.raises(PermissionError, match="Approval expired"):
        tool._run(operation="create", preview_id=preview_id)

    assert runner.calls == []


def test_approved_create_runs_preflight_and_returns_live_status(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    preview = approve_prepared(tool)

    created = json.loads(
        tool._run(operation="create", preview_id=str(preview["preview_id"]))
    )

    assert created["created"] is True
    assert created["task_id"] == TASK_ID
    assert created["task_url"] == f"https://taskmarket.dev/tasks/{TASK_ID}"
    assert created["wallet_address"] == WALLET
    assert created["maximum_spend_usdc"] == "1.250000"
    assert created["live_status"]["data"]["status"] == "open"
    assert runner.calls[:4] == [
        ("address",),
        ("deposit",),
        ("wallet", "balance"),
        ("legal", "status"),
    ]
    assert runner.calls[4] == (
        "task",
        "create",
        "--description",
        "Audit the release candidate.\n\nDeliverables:\n"
        "- Written report\n- Reproduction steps",
        "--reward",
        "1.250000",
        "--duration",
        "24",
        "--mode",
        "bounty",
        "--task-visibility",
        "public",
        "--submission-visibility",
        "winner_only",
        "--tags",
        "audit,python",
    )
    assert runner.calls[5] == ("task", "get", TASK_ID)

    with pytest.raises(PermissionError, match="already attempted"):
        tool._run(operation="create", preview_id=str(preview["preview_id"]))
    with pytest.raises(ValueError, match="already attempted"):
        tool.approve(
            str(preview["preview_id"]),
            str(preview["fingerprint_sha256"]),
        )


def test_concurrent_create_calls_cannot_duplicate_spend(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    runner.address_started = Event()
    runner.release_address = Event()
    preview = approve_prepared(tool)
    preview_id = str(preview["preview_id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(tool._run, operation="create", preview_id=preview_id)
        assert runner.address_started.wait(timeout=5)
        second = executor.submit(tool._run, operation="create", preview_id=preview_id)
        try:
            with pytest.raises(PermissionError, match="already attempted"):
                second.result(timeout=5)
        finally:
            runner.release_address.set()
        created = json.loads(first.result(timeout=5))

    assert created["created"] is True
    assert sum(call[:2] == ("task", "create") for call in runner.calls) == 1


@pytest.mark.parametrize(
    ("command", "payload", "message"),
    [
        (
            ("deposit",),
            {"ok": True, "data": {"chainId": 1, "usdcContract": BASE_USDC}},
            "not configured for Base",
        ),
        (
            ("deposit",),
            {"ok": True, "data": {"chainId": 8453, "usdcContract": "0xbad"}},
            "non-canonical Base USDC",
        ),
        (
            ("legal", "status"),
            {"ok": True, "data": {"enforcementEnabled": True, "accepted": False}},
            "requires acceptance",
        ),
        (
            ("legal", "status"),
            {"ok": True, "data": {}},
            "did not report the legal enforcement state",
        ),
        (
            ("legal", "status"),
            {"ok": True, "data": {"enforcementEnabled": True, "accepted": "false"}},
            "requires acceptance",
        ),
        (
            ("address",),
            {"ok": True, "data": {}},
            "did not return the acting wallet address",
        ),
        (
            ("address",),
            {"ok": True, "data": {"address": "   "}},
            "did not return the acting wallet address",
        ),
        (
            ("wallet", "balance"),
            {"ok": True, "data": {"balanceUsdc": "0.10"}},
            "Insufficient Base USDC",
        ),
        (
            ("wallet", "balance"),
            {"ok": True, "data": {"balanceUsdc": "Infinity"}},
            "non-finite USDC balance",
        ),
    ],
)
def test_preflight_blocks_write_when_safety_check_fails(
    tool: TaskMarketRequesterTool,
    runner: RecordedRunner,
    command: tuple[str, ...],
    payload: dict[str, object],
    message: str,
) -> None:
    runner.overrides[command] = payload
    preview = approve_prepared(tool)

    with pytest.raises(RuntimeError, match=message):
        tool._run(operation="create", preview_id=str(preview["preview_id"]))

    assert not any(call[:2] == ("task", "create") for call in runner.calls)
    with pytest.raises(PermissionError, match="already attempted"):
        tool._run(operation="create", preview_id=str(preview["preview_id"]))


def test_timed_out_write_is_unknown_and_cannot_be_retried(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    runner.create_exception = subprocess.TimeoutExpired("taskmarket", 45)
    preview = approve_prepared(tool)
    preview_id = str(preview["preview_id"])

    result = json.loads(tool._run(operation="create", preview_id=preview_id))

    assert result["created"] == "unknown"
    assert result["retry_allowed"] is False
    assert "Reconcile" in result["required_action"]
    assert sum(call[:2] == ("task", "create") for call in runner.calls) == 1
    with pytest.raises(PermissionError, match="already attempted"):
        tool._run(operation="create", preview_id=preview_id)
    assert sum(call[:2] == ("task", "create") for call in runner.calls) == 1


def test_malformed_create_success_is_unknown_and_not_retriable(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    runner.create_result = {"ok": True, "data": {"taskId": "not-a-task-id"}}
    preview = approve_prepared(tool)

    result = json.loads(
        tool._run(operation="create", preview_id=str(preview["preview_id"]))
    )

    assert result["created"] == "unknown"
    assert result["retry_allowed"] is False
    assert result["raw_result"] == runner.create_result


def test_malformed_cli_data_fails_clearly_for_reads_and_writes(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    runner.overrides[("task", "get", TASK_ID)] = {"ok": True, "data": None}
    with pytest.raises(RuntimeError, match="malformed data payload"):
        tool._run(operation="status", task_id=TASK_ID)

    runner.create_result = {"ok": True, "data": None}
    preview = approve_prepared(tool)
    result = json.loads(
        tool._run(operation="create", preview_id=str(preview["preview_id"]))
    )

    assert result["created"] == "unknown"
    assert result["retry_allowed"] is False
    assert "malformed data payload" in result["error"]


def test_status_and_submissions_are_read_only_human_review_operations(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    status = json.loads(tool._run(operation="status", task_id=TASK_ID))
    submissions = json.loads(tool._run(operation="submissions", task_id=TASK_ID))

    assert status["human_review_required"] is True
    assert submissions["human_review_required"] is True
    assert submissions["accept_or_reject_enabled"] is False
    assert runner.calls == [
        ("task", "get", TASK_ID),
        ("task", "submissions", TASK_ID),
    ]
    assert all("accept" not in call and "reject" not in call for call in runner.calls)


def test_invalid_task_id_never_reaches_cli(
    tool: TaskMarketRequesterTool, runner: RecordedRunner
) -> None:
    with pytest.raises(ValueError, match="32-byte hexadecimal"):
        tool._run(operation="status", task_id="0x123")
    assert runner.calls == []


def test_schema_rejects_unknown_operation() -> None:
    with pytest.raises(ValidationError):
        TaskMarketRequesterSchema(operation="accept")


def test_direct_run_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match="Unsupported operation"):
        TaskMarketRequesterTool()._run(operation="accept")


def test_parse_envelope_uses_last_json_line() -> None:
    parsed = TaskMarketRequesterTool._parse_envelope(
        'update available\n{"ok": true, "data": {"status": "open"}}\n',
        "",
    )

    assert parsed == {"ok": True, "data": {"status": "open"}}


def test_exported_from_package() -> None:
    from crewai_tools import TaskMarketRequesterTool as ExportedTool
    from crewai_tools.tools import TaskMarketRequesterTool as ToolsExportedTool

    assert ExportedTool is TaskMarketRequesterTool
    assert ToolsExportedTool is TaskMarketRequesterTool
