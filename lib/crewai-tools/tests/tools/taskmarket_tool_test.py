import json
from urllib.request import Request

from crewai_tools import TaskMarketGetTaskTool, TaskMarketSearchTool


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._payload


def test_search_filters_open_tasks_by_usdc_ceiling(monkeypatch):
    requests = []

    def fake_urlopen(request: Request, timeout: int):
        requests.append((request.full_url, timeout))
        return FakeResponse(
            {
                "tasks": [
                    {"id": "cheap", "reward": "500000", "status": "open", "tags": ["python"]},
                    {"id": "expensive", "reward": "1000001", "status": "open"},
                    {"id": "closed", "reward": "1", "status": "closed"},
                ]
            }
        )

    monkeypatch.setattr("crewai_tools.tools.taskmarket_tool.taskmarket_tool.urlopen", fake_urlopen)
    result = json.loads(TaskMarketSearchTool()._run(max_reward_usdc="1", tags="python"))

    assert [row["id"] for row in result] == ["cheap"]
    assert result[0]["reward_usdc"] == "0.5"
    assert "status=open" in requests[0][0]
    assert "tags=python" in requests[0][0]


def test_search_respects_limit_before_processing_rows(monkeypatch):
    def fake_urlopen(_request: Request, timeout: int):
        assert timeout == 20
        return FakeResponse(
            {
                "tasks": [
                    {"id": "first", "reward": "100000", "status": "open"},
                    {"id": "second", "reward": "200000", "status": "open"},
                    {"id": "third", "reward": "300000", "status": "open"},
                ]
            }
        )

    monkeypatch.setattr("crewai_tools.tools.taskmarket_tool.taskmarket_tool.urlopen", fake_urlopen)
    result = json.loads(TaskMarketSearchTool()._run(max_reward_usdc="1", limit=2))

    assert [row["id"] for row in result] == ["first", "second"]


def test_get_task_is_read_only_and_normalizes_decimal_reward(monkeypatch):
    requests = []

    def fake_urlopen(request: Request, timeout: int):
        requests.append((request.method, request.full_url))
        return FakeResponse(
            {
                "id": "task-1",
                "rewardUsdc": "0.25",
                "status": "open",
                "description": "A public task",
            }
        )

    monkeypatch.setattr("crewai_tools.tools.taskmarket_tool.taskmarket_tool.urlopen", fake_urlopen)
    result = json.loads(TaskMarketGetTaskTool()._run("task-1"))

    assert result["reward_usdc"] == "0.25"
    assert requests == [("GET", "https://api.taskmarket.dev/api/tasks/task-1")]


def test_get_task_rejects_path_traversal():
    for task_id in ("../wallet", ".", ".."):
        try:
            TaskMarketGetTaskTool()._run(task_id)
        except ValueError as exc:
            assert "opaque ID" in str(exc)
        else:
            raise AssertionError(f"unsafe task ID should be rejected: {task_id!r}")


def test_get_task_encodes_query_fragment_and_encoded_separator(monkeypatch):
    requests = []

    def fake_urlopen(request: Request, timeout: int):
        requests.append((request.method, request.full_url, timeout))
        return FakeResponse({"id": "safe", "rewardUsdc": "0.1", "status": "open"})

    monkeypatch.setattr("crewai_tools.tools.taskmarket_tool.taskmarket_tool.urlopen", fake_urlopen)
    for task_id in ("task?query", "task#fragment", "task%2Fencoded"):
        TaskMarketGetTaskTool()._run(task_id)

    assert [url for _method, url, _timeout in requests] == [
        "https://api.taskmarket.dev/api/tasks/task%3Fquery",
        "https://api.taskmarket.dev/api/tasks/task%23fragment",
        "https://api.taskmarket.dev/api/tasks/task%252Fencoded",
    ]
