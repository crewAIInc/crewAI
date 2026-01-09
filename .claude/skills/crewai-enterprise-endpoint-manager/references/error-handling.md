# Error Handling Patterns for CrewAI Enterprise API

Robust error handling strategies for production deployments.

## Contents

- [Error Types](#error-types) - HTTP status codes and execution errors
- [Retry Strategies](#retry-strategies) - Exponential backoff, selective retry
- [Rate Limit Handling](#rate-limit-handling) - Detect limits, adaptive concurrency
- [Circuit Breaker Pattern](#circuit-breaker-pattern) - Prevent cascading failures
- [Error Aggregation](#error-aggregation-and-reporting) - Collect and summarize batch errors
- [Timeout Handling](#timeout-handling) - Separate execution and poll timeouts
- [Resilient Batch Execution](#complete-error-resilient-batch-execution) - Full pattern with auto-retry

## Error Types

### HTTP Status Errors

| Status Code | Meaning | Action |
|-------------|---------|--------|
| 400 | Bad Request | Check input format, fix and retry |
| 401 | Unauthorized | Token invalid/expired, refresh token |
| 403 | Forbidden | No access to this crew |
| 404 | Not Found | Crew URL or kickoff_id doesn't exist |
| 429 | Rate Limited | Implement backoff, reduce concurrency |
| 500 | Server Error | Retry with exponential backoff |
| 502/503 | Service Unavailable | Wait and retry |

### Execution Errors

```json
{
  "status": "error",
  "error_message": "Failed to complete task due to invalid input."
}
```

Common causes:
- Invalid inputs for the crew's expected schema
- Agent task failures
- LLM API errors (rate limits, timeouts)
- Tool execution failures

## Retry Strategies

### Exponential Backoff

```python
import asyncio
import random

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
):
    """
    Retry with exponential backoff and optional jitter.

    Args:
        func: Async function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter: Add randomness to prevent thundering herd
    """
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())

            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
```

### Selective Retry

```python
from httpx import HTTPStatusError

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

async def selective_retry(func, max_retries: int = 3):
    """Only retry on specific error types."""
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except HTTPStatusError as e:
            if e.response.status_code not in RETRYABLE_STATUS_CODES:
                raise  # Don't retry 400, 401, 403, 404
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 ** attempt)
        except (asyncio.TimeoutError, ConnectionError) as e:
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 ** attempt)
```

## Rate Limit Handling

### Detect and Respect Rate Limits

```python
import asyncio
from datetime import datetime, timedelta

class RateLimitHandler:
    """Handle 429 responses with retry-after headers."""

    def __init__(self):
        self.blocked_until: datetime | None = None

    async def wait_if_blocked(self):
        """Wait if currently rate limited."""
        if self.blocked_until and datetime.now() < self.blocked_until:
            wait_time = (self.blocked_until - datetime.now()).total_seconds()
            print(f"Rate limited. Waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

    def handle_response(self, response):
        """Check response for rate limit headers."""
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            try:
                seconds = int(retry_after)
            except ValueError:
                seconds = 60

            self.blocked_until = datetime.now() + timedelta(seconds=seconds)
            raise RateLimitError(f"Rate limited for {seconds}s")

class RateLimitError(Exception):
    pass
```

### Adaptive Concurrency

```python
class AdaptiveSemaphore:
    """
    Automatically reduce concurrency on rate limits.
    """

    def __init__(self, initial: int = 10, minimum: int = 1):
        self.current = initial
        self.minimum = minimum
        self.semaphore = asyncio.Semaphore(initial)
        self.lock = asyncio.Lock()
        self._rate_limit_count = 0

    async def acquire(self):
        await self.semaphore.acquire()

    def release(self):
        self.semaphore.release()

    async def reduce_on_rate_limit(self):
        """Call when hitting rate limit."""
        async with self.lock:
            self._rate_limit_count += 1
            if self._rate_limit_count >= 3 and self.current > self.minimum:
                new_value = max(self.current // 2, self.minimum)
                print(f"Reducing concurrency: {self.current} -> {new_value}")
                self.current = new_value
                self.semaphore = asyncio.Semaphore(new_value)
                self._rate_limit_count = 0
```

## Circuit Breaker Pattern

Prevent cascading failures by stopping requests when error rate is too high.

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_time=30)

        async def make_request():
            if not breaker.allow_request():
                raise CircuitOpenError("Circuit is open")

            try:
                result = await client.kickoff(inputs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
    """
    failure_threshold: int = 5
    recovery_time: float = 30.0  # seconds

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: datetime | None = None

    def allow_request(self) -> bool:
        """Check if request is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        # HALF_OPEN: allow one request to test
        return True

    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"Circuit opened after {self.failure_count} failures")

    def _should_attempt_recovery(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_time


class CircuitOpenError(Exception):
    pass
```

## Error Aggregation and Reporting

```python
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ErrorAggregator:
    """Collect and summarize errors from batch executions."""

    errors: list[tuple[dict, str]] = field(default_factory=list)
    error_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_error(self, inputs: dict, error_message: str):
        """Record an error."""
        self.errors.append((inputs, error_message))

        # Categorize error
        category = self._categorize(error_message)
        self.error_counts[category] += 1

    def _categorize(self, error_message: str) -> str:
        """Categorize error for aggregation."""
        lower = error_message.lower()
        if "rate limit" in lower or "429" in lower:
            return "rate_limit"
        elif "timeout" in lower:
            return "timeout"
        elif "401" in lower or "unauthorized" in lower:
            return "auth"
        elif "invalid input" in lower:
            return "invalid_input"
        elif "500" in lower or "server error" in lower:
            return "server_error"
        return "other"

    def get_summary(self) -> dict:
        """Get error summary."""
        return {
            "total_errors": len(self.errors),
            "by_category": dict(self.error_counts),
            "sample_errors": self.errors[:5]  # First 5 for inspection
        }

    def get_retryable_inputs(self) -> list[dict]:
        """Get inputs that failed with retryable errors."""
        retryable_categories = {"rate_limit", "timeout", "server_error"}
        return [
            inputs for inputs, error in self.errors
            if self._categorize(error) in retryable_categories
        ]


# Usage
aggregator = ErrorAggregator()

for result in batch_results:
    if result.error:
        aggregator.add_error(result.inputs, result.error)

summary = aggregator.get_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"By category: {summary['by_category']}")

# Retry failed inputs
retryable = aggregator.get_retryable_inputs()
if retryable:
    print(f"Retrying {len(retryable)} failed executions...")
    retry_results = await client.kickoff_batch(retryable, max_concurrent=5)
```

## Timeout Handling

```python
async def kickoff_with_timeout(
    client,
    inputs: dict,
    execution_timeout: float = 600.0,
    poll_timeout: float = 10.0
) -> ExecutionResult:
    """
    Kickoff with separate timeouts for execution and polling.

    Args:
        execution_timeout: Max total time for the crew to complete
        poll_timeout: Timeout for each status poll request
    """
    start_time = datetime.now()

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        # Kickoff (short timeout)
        resp = await http_client.post(
            f"{client.base_url}/kickoff",
            headers=client._headers,
            json={"inputs": inputs}
        )
        kickoff_id = resp.json()["kickoff_id"]

        # Poll with timeout
        while True:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > execution_timeout:
                return ExecutionResult(
                    kickoff_id=kickoff_id,
                    inputs=inputs,
                    status=ExecutionStatus.ERROR,
                    error=f"Execution timeout after {elapsed:.0f}s"
                )

            try:
                async with asyncio.timeout(poll_timeout):
                    resp = await http_client.get(
                        f"{client.base_url}/{kickoff_id}/status",
                        headers=client._headers
                    )
            except asyncio.TimeoutError:
                continue  # Retry poll

            data = resp.json()
            if data["status"] in ("completed", "error"):
                return ExecutionResult(...)

            await asyncio.sleep(2)
```

## Complete Error-Resilient Batch Execution

```python
async def resilient_batch_execution(
    client: CrewAIClient,
    inputs_list: list[dict],
    max_concurrent: int = 10,
    max_total_retries: int = 3
) -> tuple[list[ExecutionResult], list[ExecutionResult]]:
    """
    Execute batch with automatic retry of failed executions.

    Returns:
        Tuple of (successful_results, final_failed_results)
    """
    all_successful = []
    remaining = inputs_list.copy()

    for retry_round in range(max_total_retries + 1):
        if not remaining:
            break

        if retry_round > 0:
            print(f"Retry round {retry_round}: {len(remaining)} executions")
            # Reduce concurrency on retries
            current_concurrent = max(max_concurrent // (2 ** retry_round), 1)
        else:
            current_concurrent = max_concurrent

        results = await client.kickoff_batch(
            remaining,
            max_concurrent=current_concurrent
        )

        # Separate successful and failed
        successful = [r for r in results if r.is_success]
        failed = [r for r in results if not r.is_success]

        all_successful.extend(successful)

        # Get retryable failures
        aggregator = ErrorAggregator()
        for r in failed:
            aggregator.add_error(r.inputs, r.error or "Unknown error")

        remaining = aggregator.get_retryable_inputs()

        if not remaining:
            # All remaining failures are non-retryable
            return all_successful, failed

        await asyncio.sleep(5 * (retry_round + 1))  # Increasing wait between rounds

    # Return final state
    final_failed = [
        ExecutionResult(
            kickoff_id="",
            inputs=inputs,
            status=ExecutionStatus.ERROR,
            error="Max retries exceeded"
        )
        for inputs in remaining
    ]

    return all_successful, final_failed
```
