# CrewAI Enterprise Python Client

Complete Python client implementation for interacting with deployed CrewAI crews and flows.

## Contents

- [Full-Featured Client Class](#full-featured-client-class) - Production-ready `CrewAIClient` with sync/async methods
- [Usage Examples](#usage-examples) - Basic, async, structured output, error handling
- [Environment Configuration](#environment-configuration) - Config from environment variables
- [Rate-Limited Semaphore](#rate-limited-semaphore) - Combine concurrency with rate limiting
- [Quick Start Examples](#quick-start-examples) - Minimal code to get started
- [Dependencies](#dependencies) - Required packages

## Full-Featured Client Class

```python
"""
CrewAI Enterprise API Client

Production-ready client for interacting with deployed crews and flows.
Supports synchronous, async, and batch operations with semaphore control.
"""

import asyncio
import httpx
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TypeVar, Generic
from enum import Enum
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of a single crew execution."""
    kickoff_id: str
    inputs: dict
    status: ExecutionStatus
    result: dict | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: dict | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED and self.result is not None


@dataclass
class CrewAIClient:
    """
    Client for CrewAI Enterprise API.

    Usage:
        client = CrewAIClient(
            base_url="https://your-crew.crewai.com",
            token="YOUR_TOKEN"
        )

        # Sync usage
        result = client.kickoff_sync({"topic": "AI"})

        # Async usage
        result = await client.kickoff({"topic": "AI"})

        # Batch with semaphore
        results = await client.kickoff_batch(inputs_list, max_concurrent=10)
    """
    base_url: str
    token: str
    timeout: float = 600.0
    poll_interval: float = 2.0
    max_retries: int = 3

    _headers: dict = field(init=False)

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    # =========================================================================
    # Input Discovery
    # =========================================================================

    async def get_inputs(self) -> dict:
        """Discover required inputs for the crew."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.base_url}/inputs",
                headers=self._headers
            )
            resp.raise_for_status()
            return resp.json()

    def get_inputs_sync(self) -> dict:
        """Synchronous version of get_inputs."""
        import requests
        resp = requests.get(
            f"{self.base_url}/inputs",
            headers=self._headers,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    # =========================================================================
    # Single Execution
    # =========================================================================

    async def kickoff(
        self,
        inputs: dict,
        wait: bool = True,
        webhook_config: dict | None = None
    ) -> ExecutionResult:
        """
        Start a crew execution.

        Args:
            inputs: Input parameters for the crew
            wait: If True, poll until completion
            webhook_config: Optional HITL webhook configuration

        Returns:
            ExecutionResult with status and results
        """
        started_at = datetime.now()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Build request body
            body: dict[str, Any] = {"inputs": inputs}
            if webhook_config:
                body["humanInputWebhook"] = webhook_config

            # Kickoff
            resp = await client.post(
                f"{self.base_url}/kickoff",
                headers=self._headers,
                json=body
            )
            resp.raise_for_status()
            kickoff_id = resp.json()["kickoff_id"]

            if not wait:
                return ExecutionResult(
                    kickoff_id=kickoff_id,
                    inputs=inputs,
                    status=ExecutionStatus.RUNNING,
                    started_at=started_at
                )

            # Poll for completion
            return await self._poll_until_complete(
                client, kickoff_id, inputs, started_at
            )

    def kickoff_sync(
        self,
        inputs: dict,
        wait: bool = True,
        webhook_config: dict | None = None
    ) -> ExecutionResult:
        """Synchronous version of kickoff."""
        return asyncio.run(self.kickoff(inputs, wait, webhook_config))

    async def get_status(self, kickoff_id: str) -> ExecutionResult:
        """Get the current status of an execution."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.base_url}/{kickoff_id}/status",
                headers=self._headers
            )
            resp.raise_for_status()
            data = resp.json()

            return ExecutionResult(
                kickoff_id=kickoff_id,
                inputs={},
                status=ExecutionStatus(data["status"]),
                result=data.get("result"),
                error=data.get("error_message"),
                progress=data.get("progress")
            )

    # =========================================================================
    # Batch Execution with Semaphore
    # =========================================================================

    async def kickoff_batch(
        self,
        inputs_list: list[dict],
        max_concurrent: int = 10,
        on_progress: Callable[[int, int, ExecutionResult], None] | None = None,
        on_error: Callable[[ExecutionResult], bool] | None = None
    ) -> list[ExecutionResult]:
        """
        Execute multiple crews with semaphore-controlled concurrency.

        Args:
            inputs_list: List of input dictionaries
            max_concurrent: Maximum concurrent executions
            on_progress: Callback(completed, total, result) for progress
            on_error: Callback(result) -> bool, return True to continue

        Returns:
            List of ExecutionResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[ExecutionResult] = []
        completed_count = 0
        total = len(inputs_list)
        lock = asyncio.Lock()

        async def execute_with_semaphore(inputs: dict) -> ExecutionResult:
            nonlocal completed_count

            async with semaphore:
                result = await self._execute_single_with_retry(inputs)

            async with lock:
                completed_count += 1
                results.append(result)

                if on_progress:
                    on_progress(completed_count, total, result)

                if result.error and on_error:
                    should_continue = on_error(result)
                    if not should_continue:
                        raise asyncio.CancelledError("Stopped by on_error callback")

                logger.info(
                    f"[{completed_count}/{total}] "
                    f"{'OK' if result.is_success else 'ERR'} "
                    f"id={result.kickoff_id}"
                )

            return result

        tasks = [execute_with_semaphore(inputs) for inputs in inputs_list]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.warning("Batch execution cancelled")

        return results

    def kickoff_batch_sync(
        self,
        inputs_list: list[dict],
        max_concurrent: int = 10,
        on_progress: Callable[[int, int, ExecutionResult], None] | None = None
    ) -> list[ExecutionResult]:
        """Synchronous version of kickoff_batch."""
        return asyncio.run(
            self.kickoff_batch(inputs_list, max_concurrent, on_progress)
        )

    # =========================================================================
    # Structured Output Support
    # =========================================================================

    async def kickoff_typed(
        self,
        inputs: dict,
        output_model: type[T]
    ) -> T | None:
        """
        Execute and parse result into a Pydantic model.

        Args:
            inputs: Input parameters
            output_model: Pydantic model class for parsing

        Returns:
            Parsed model instance or None if failed
        """
        result = await self.kickoff(inputs, wait=True)

        if result.is_success and result.result:
            output = result.result.get("output", "")
            # Try to parse JSON from output
            import json
            try:
                data = json.loads(output)
                return output_model.model_validate(data)
            except (json.JSONDecodeError, Exception):
                return None
        return None

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _execute_single_with_retry(self, inputs: dict) -> ExecutionResult:
        """Execute single crew with retry logic."""
        started_at = datetime.now()

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Kickoff
                    resp = await client.post(
                        f"{self.base_url}/kickoff",
                        headers=self._headers,
                        json={"inputs": inputs}
                    )
                    resp.raise_for_status()
                    kickoff_id = resp.json()["kickoff_id"]

                    # Poll
                    return await self._poll_until_complete(
                        client, kickoff_id, inputs, started_at
                    )

            except httpx.HTTPStatusError as e:
                if attempt == self.max_retries - 1:
                    return ExecutionResult(
                        kickoff_id="",
                        inputs=inputs,
                        status=ExecutionStatus.ERROR,
                        error=f"HTTP {e.response.status_code}: {str(e)}",
                        started_at=started_at,
                        completed_at=datetime.now()
                    )
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return ExecutionResult(
                        kickoff_id="",
                        inputs=inputs,
                        status=ExecutionStatus.ERROR,
                        error=str(e),
                        started_at=started_at,
                        completed_at=datetime.now()
                    )
                await asyncio.sleep(2 ** attempt)

        # Should never reach here
        return ExecutionResult(
            kickoff_id="",
            inputs=inputs,
            status=ExecutionStatus.ERROR,
            error="Max retries exceeded",
            started_at=started_at,
            completed_at=datetime.now()
        )

    async def _poll_until_complete(
        self,
        client: httpx.AsyncClient,
        kickoff_id: str,
        inputs: dict,
        started_at: datetime
    ) -> ExecutionResult:
        """Poll status until completion or error."""
        while True:
            resp = await client.get(
                f"{self.base_url}/{kickoff_id}/status",
                headers=self._headers
            )
            resp.raise_for_status()
            data = resp.json()

            status = ExecutionStatus(data["status"])

            if status == ExecutionStatus.COMPLETED:
                return ExecutionResult(
                    kickoff_id=kickoff_id,
                    inputs=inputs,
                    status=status,
                    result=data.get("result"),
                    started_at=started_at,
                    completed_at=datetime.now()
                )

            if status == ExecutionStatus.ERROR:
                return ExecutionResult(
                    kickoff_id=kickoff_id,
                    inputs=inputs,
                    status=status,
                    error=data.get("error_message", "Unknown error"),
                    started_at=started_at,
                    completed_at=datetime.now()
                )

            await asyncio.sleep(self.poll_interval)
```

## Usage Examples

### Basic Usage

```python
# Initialize client
client = CrewAIClient(
    base_url="https://your-crew.crewai.com",
    token="YOUR_TOKEN"
)

# Discover inputs
inputs_schema = client.get_inputs_sync()
print(f"Required inputs: {inputs_schema}")

# Single execution (sync)
result = client.kickoff_sync({"topic": "AI Research"})
if result.is_success:
    print(f"Output: {result.result['output']}")
else:
    print(f"Error: {result.error}")
```

### Async with Progress Tracking

```python
import asyncio

async def main():
    client = CrewAIClient(
        base_url="https://your-crew.crewai.com",
        token="YOUR_TOKEN",
        max_concurrent=10
    )

    # 100 executions
    inputs_list = [{"topic": f"Topic {i}"} for i in range(100)]

    def on_progress(completed, total, result):
        pct = completed / total * 100
        status = "OK" if result.is_success else "ERR"
        print(f"[{completed}/{total}] {pct:.0f}% - {status}")

    results = await client.kickoff_batch(
        inputs_list,
        max_concurrent=10,
        on_progress=on_progress
    )

    # Summary
    success = sum(1 for r in results if r.is_success)
    print(f"\nCompleted: {success}/{len(results)}")

asyncio.run(main())
```

### With Structured Output

```python
from pydantic import BaseModel

class ResearchOutput(BaseModel):
    summary: str
    key_findings: list[str]
    confidence: float

async def main():
    client = CrewAIClient(...)

    output = await client.kickoff_typed(
        inputs={"topic": "Quantum Computing"},
        output_model=ResearchOutput
    )

    if output:
        print(f"Summary: {output.summary}")
        print(f"Findings: {output.key_findings}")
```

### Error Handling with Early Stop

```python
async def main():
    client = CrewAIClient(...)

    error_count = 0
    max_errors = 5

    def on_error(result):
        nonlocal error_count
        error_count += 1
        print(f"Error {error_count}: {result.error}")
        # Stop if too many errors
        return error_count < max_errors

    results = await client.kickoff_batch(
        inputs_list,
        max_concurrent=10,
        on_error=on_error
    )
```

## Environment Configuration

```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    base_url: str = os.getenv("CREWAI_BASE_URL", "")
    token: str = os.getenv("CREWAI_TOKEN", "")
    max_concurrent: int = int(os.getenv("CREWAI_MAX_CONCURRENT", "10"))
    timeout: float = float(os.getenv("CREWAI_TIMEOUT", "600"))

config = Config()
client = CrewAIClient(
    base_url=config.base_url,
    token=config.token,
    timeout=config.timeout
)
```

## Rate-Limited Semaphore

For APIs with rate limits, combine concurrency control with rate limiting:

```python
import asyncio
from collections import deque
from time import time

class RateLimitedSemaphore:
    """
    Semaphore with rate limiting: max N concurrent requests AND max M requests per second.

    Usage:
        limiter = RateLimitedSemaphore(max_concurrent=10, max_per_second=5.0)

        async def make_request():
            async with limiter:
                return await client.kickoff(inputs)
    """

    def __init__(self, max_concurrent: int, max_per_second: float):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_per_second = max_per_second
        self.request_times: deque = deque()
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        await self.semaphore.acquire()

        async with self.lock:
            now = time()
            # Remove timestamps older than 1 second
            while self.request_times and now - self.request_times[0] > 1.0:
                self.request_times.popleft()

            # If at rate limit, wait
            if len(self.request_times) >= self.max_per_second:
                sleep_time = 1.0 - (now - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self.request_times.append(time())

    async def __aexit__(self, *args):
        self.semaphore.release()


# Usage with CrewAIClient
async def batch_with_rate_limit(client: CrewAIClient, inputs_list: list[dict]):
    """Execute batch with both concurrency and rate limiting."""
    limiter = RateLimitedSemaphore(max_concurrent=10, max_per_second=5.0)

    async def single_execution(inputs: dict) -> ExecutionResult:
        async with limiter:
            return await client.kickoff(inputs, wait=True)

    tasks = [single_execution(inputs) for inputs in inputs_list]
    return await asyncio.gather(*tasks)
```

## Quick Start Examples

### Minimal Example

```python
# Simplest possible usage
client = CrewAIClient(
    base_url="https://your-crew.crewai.com",
    token="YOUR_TOKEN"
)

result = client.kickoff_sync({"topic": "AI Research"})
print(result.result["output"] if result.is_success else result.error)
```

### Batch Processing 100 Items

```python
import asyncio

async def process_batch():
    client = CrewAIClient(
        base_url="https://your-crew.crewai.com",
        token="YOUR_TOKEN"
    )

    inputs = [{"topic": f"Topic {i}"} for i in range(100)]

    results = await client.kickoff_batch(
        inputs,
        max_concurrent=10,
        on_progress=lambda done, total, r: print(f"{done}/{total}")
    )

    success = sum(1 for r in results if r.is_success)
    print(f"Success: {success}/{len(results)}")

asyncio.run(process_batch())
```

## Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
]
```
