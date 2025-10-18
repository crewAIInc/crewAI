from typing import Any


class LongTermMemoryItem:
    def __init__(
        self,
        agent: str,
        task: str,
        expected_output: str,
        datetime: str,
        quality: int | float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.task = task
        self.agent = agent
        self.quality = quality
        self.datetime = datetime
        self.expected_output = expected_output
        self.metadata = metadata if metadata is not None else {}
