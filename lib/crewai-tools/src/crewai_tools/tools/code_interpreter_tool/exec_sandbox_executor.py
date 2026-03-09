"""Exec-sandbox executor for running code in QEMU microVMs.

This module provides integration with exec-sandbox (https://github.com/dualeai/exec-sandbox)
for executing Python code in hardware-isolated QEMU microVMs. This provides stronger
isolation than Docker containers without requiring a Docker daemon.

Key benefits:
- Hardware-level isolation (KVM/HVF) vs container isolation
- No Docker daemon required - works in CI/CD and restricted environments
- No Docker Desktop licensing costs for enterprises
- Fresh VM per execution with no state leakage
- Fast warm starts (1-2ms from pre-booted pool)
- Built-in network control and file I/O isolation
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class ExecSandboxExecutor:
    """Executor for running Python code in exec-sandbox microVMs.

    This class wraps the exec-sandbox library to provide code execution
    in hardware-isolated QEMU microVMs. Each execution gets a fresh VM
    that is destroyed after completion, ensuring no state leakage.

    Example:
        ```python
        executor = ExecSandboxExecutor()
        result = await executor.execute(
            code="print('Hello from microVM!')",
            packages=["requests"],
            timeout_seconds=30
        )
        print(result.stdout)
        ```

    References:
        - exec-sandbox GitHub: https://github.com/dualeai/exec-sandbox
        - exec-sandbox PyPI: https://pypi.org/project/exec-sandbox/
    """

    async def execute(
        self,
        code: str,
        packages: list[str] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        """Execute Python code in a QEMU microVM sandbox.

        Args:
            code: Python 3 code to execute.
            packages: List of pip package names to install before execution.
            timeout_seconds: Maximum execution time in seconds.

        Returns:
            Dictionary with execution results:
                - exit_code: int, 0 on success
                - stdout: str, standard output
                - stderr: str, standard error
                - execution_time_ms: float, execution duration

        Raises:
            ImportError: If exec-sandbox package is not installed.
            RuntimeError: If execution fails.

        Note:
            Each execution runs in a fresh microVM that is destroyed after
            completion. For better performance with multiple executions,
            consider using the session-based API for stateful workflows.
        """
        try:
            from exec_sandbox import AsyncScheduler
        except ImportError as e:
            raise ImportError(
                "exec-sandbox is not installed. "
                "Install it with: pip install exec-sandbox"
            ) from e

        async with AsyncScheduler() as scheduler:
            result = await scheduler.run(
                code=code,
                language="python",
                packages=packages or [],
                timeout_seconds=timeout_seconds,
            )

            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time_ms": result.execution_time_ms,
            }

    def execute_sync(
        self,
        code: str,
        packages: list[str] | None = None,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        """Synchronously execute Python code in a QEMU microVM sandbox.

        This is a convenience wrapper around the async execute() method
        for use in synchronous contexts.

        Args:
            code: Python 3 code to execute.
            packages: List of pip package names to install before execution.
            timeout_seconds: Maximum execution time in seconds.

        Returns:
            Dictionary with execution results (see execute() for details).

        Raises:
            ImportError: If exec-sandbox package is not installed.
            RuntimeError: If execution fails.

        Note:
            If called from within an existing event loop, uses ThreadPoolExecutor
            to avoid RuntimeError from asyncio.run().
        """
        try:
            # Check if we're already in an event loop (e.g., called from async context)
            asyncio.get_running_loop()
            # If we get here, a loop is running - use executor instead
            with ThreadPoolExecutor() as executor:
                return executor.submit(
                    asyncio.run, self.execute(code, packages, timeout_seconds)
                ).result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self.execute(code, packages, timeout_seconds))
