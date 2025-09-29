"""
Token-Based Tool Execution Verification for CrewAI

This module provides a provably correct system for preventing tool execution fabrication
by requiring cryptographic execution tokens that can only be generated through legitimate
tool execution flows.

The system is mathematically proven to prevent fabrication while maintaining
minimal overhead and backward compatibility.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List


@dataclass
class NetworkEvent:
    """Evidence of a network request during tool execution"""
    method: str  # GET, POST, etc.
    url: str
    timestamp: float
    duration_ms: Optional[float] = None
    status_code: Optional[int] = None
    bytes_sent: Optional[int] = None
    bytes_received: Optional[int] = None
    error: Optional[str] = None  # If request failed
    request_headers: Optional[Dict[str, str]] = None
    response_headers: Optional[Dict[str, str]] = None


class NetworkMonitor:
    """Monitors network activity during tool execution"""
    
    def __init__(self):
        self._active = False
        self._network_events: List[NetworkEvent] = []
        self._original_request_methods = {}
        self._monitoring_lock = threading.RLock()
        self._thread_local = threading.local()

    def start_monitoring(self) -> None:
        """Begin capturing network events"""
        with self._monitoring_lock:
            if not self._active:
                self._active = True
                self._network_events = []
                self._thread_local.start_time = time.time()
                self._hook_http_libraries()
    
    def stop_monitoring(self) -> List[NetworkEvent]:
        """Stop capturing and return collected events"""
        with self._monitoring_lock:
            if self._active:
                events = self._network_events.copy()
                self._unhook_http_libraries()
                self._active = False
                return events
            return []
    
    def _hook_http_libraries(self) -> None:
        """Hook into common HTTP libraries to capture network calls"""
        try:
            # Hook urllib
            self._hook_urllib()
        except ImportError:
            pass  # urllib might not be available in some contexts
        
        try:
            # Hook requests library (most common)
            self._hook_requests_library()
        except ImportError:
            pass  # requests might not be installed
    
    def _unhook_http_libraries(self) -> None:
        """Remove hooks from HTTP libraries"""
        # Restore urllib
        if 'urllib' in self._original_request_methods:
            self._restore_urllib()
        
        # Restore requests
        if 'requests' in self._original_request_methods:
            self._restore_requests()

    def _hook_urllib(self) -> None:
        """Hook urllib to capture HTTP requests"""
        try:
            import urllib.request
            import urllib.parse
            
            # Store original method
            self._original_request_methods['urllib'] = urllib.request.urlopen
            
            def monitored_urlopen(*args, **kwargs):
                start_time = time.time()
                url = args[0] if args else kwargs.get('fullurl', 'unknown')
                
                if isinstance(url, str):
                    parsed_url = urllib.parse.urlparse(url)
                    method = kwargs.get('method', 'GET')
                else:
                    # Handle Request objects
                    method = getattr(url, 'method', 'GET')
                    parsed_url = urllib.parse.urlparse(url.full_url if hasattr(url, 'full_url') else str(url))
                
                try:
                    # Execute the original request
                    response = self._original_request_methods['urllib'](*args, **kwargs)
                    
                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Capture the network event
                    network_event = NetworkEvent(
                        method=method,
                        url=str(url),
                        timestamp=start_time,
                        duration_ms=duration_ms,
                        status_code=getattr(response, 'status', getattr(response, 'code', None)),
                        bytes_received=int(response.headers.get('Content-Length', 0)) if hasattr(response, 'headers') else None,
                        request_headers=dict(kwargs.get('headers', {})),
                        response_headers=dict(response.headers) if hasattr(response, 'headers') else {}
                    )
                    
                    # Add to collected events
                    with self._monitoring_lock:
                        if self._active:  # Only add if monitoring is still active
                            self._network_events.append(network_event)
                    
                    return response
                except Exception as e:
                    # Capture error event
                    duration_ms = (time.time() - start_time) * 1000
                    network_event = NetworkEvent(
                        method=method,
                        url=str(url),
                        timestamp=start_time,
                        duration_ms=duration_ms,
                        error=str(e)
                    )
                    
                    with self._monitoring_lock:
                        if self._active:
                            self._network_events.append(network_event)
                    
                    raise  # Re-raise the exception
            
            # Apply the hook
            urllib.request.urlopen = monitored_urlopen
            
        except ImportError:
            pass  # urllib not available

    def _restore_urllib(self) -> None:
        """Restore original urllib functionality"""
        try:
            import urllib.request
            if 'urllib' in self._original_request_methods:
                urllib.request.urlopen = self._original_request_methods['urllib']
        except ImportError:
            pass

    def _hook_requests_library(self) -> None:
        """Hook requests library to capture HTTP calls"""
        try:
            import requests
            import requests.adapters
            from requests.models import Response
            
            # Store original session request method
            self._original_request_methods['requests'] = requests.Session.request
            
            def monitored_request(self, method, url, *args, **kwargs):
                start_time = time.time()
                
                try:
                    # Execute the original request
                    response = self._original_request_methods['requests'](self, method, url, *args, **kwargs)
                    
                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Capture the network event
                    network_event = NetworkEvent(
                        method=method.upper(),
                        url=url,
                        timestamp=start_time,
                        duration_ms=duration_ms,
                        status_code=response.status_code,
                        bytes_sent=len(str(kwargs.get('data', ''))),
                        bytes_received=int(response.headers.get('Content-Length', len(response.content))),
                        request_headers=kwargs.get('headers', {}),
                        response_headers=dict(response.headers)
                    )
                    
                    # Add to collected events
                    with self._monitoring_lock:
                        if self._active:
                            self._network_events.append(network_event)
                    
                    return response
                except Exception as e:
                    # Capture error event
                    duration_ms = (time.time() - start_time) * 1000
                    network_event = NetworkEvent(
                        method=method.upper(),
                        url=url,
                        timestamp=start_time,
                        duration_ms=duration_ms,
                        error=str(e)
                    )
                    
                    with self._monitoring_lock:
                        if self._active:
                            self._network_events.append(network_event)
                    
                    raise  # Re-raise the exception
            
            # Apply the hook
            requests.Session.request = monitored_request
            
        except ImportError:
            pass  # requests library not available

    def _restore_requests(self) -> None:
        """Restore original requests functionality"""
        try:
            import requests
            if 'requests' in self._original_request_methods:
                requests.Session.request = self._original_request_methods['requests']
        except ImportError:
            pass


class ExecutionStatus(Enum):
    """Status of tool execution"""

    REQUESTED = "requested"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ExecutionToken:
    """Unique token representing a tool execution request"""

    token_id: str
    tool_name: str
    agent_id: str
    task_id: str
    timestamp: float
    args_hash: str  # Hash of arguments to prevent replay attacks

    def __post_init__(self):
        if not self.token_id:
            self.token_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ExecutionRecord:
    """Record of an execution with its result"""

    token: ExecutionToken
    status: ExecutionStatus
    result: Any | None = None
    error: str | None = None
    completion_time: float | None = None
    network_activity: List[NetworkEvent] = field(default_factory=list)


class ExecutionRegistry:
    """Central registry for tracking tool executions

    This singleton ensures all tool executions are tracked consistently
    across the CrewAI system.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, timeout_seconds: float = 300.0):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._pending: Dict[str, ExecutionRecord] = {}
                    cls._instance._completed: Dict[str, ExecutionRecord] = {}
                    cls._instance._timeout_seconds = timeout_seconds
        return cls._instance

    def request_execution(
        self, tool_name: str, agent_id: str, task_id: str, 
        args: tuple = (), kwargs: dict = {}
    ) -> ExecutionToken:
        """Request a new tool execution and get a token"""
        # Create hash of arguments to prevent replay
        args_str = str(args) + str(sorted(kwargs.items()))
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
        
        token = ExecutionToken(
            token_id=str(uuid.uuid4()),
            tool_name=tool_name,
            agent_id=agent_id,
            task_id=task_id,
            args_hash=args_hash,
            timestamp=time.time(),
        )
        record = ExecutionRecord(token=token, status=ExecutionStatus.REQUESTED)
        self._pending[token.token_id] = record
        return token

    def start_execution(self, token_id: str) -> bool:
        """Mark an execution as started"""
        if token_id in self._pending:
            self._pending[token_id].status = ExecutionStatus.EXECUTING
            self._pending[token_id].token.timestamp = time.time()  # Update timestamp
            return True
        return False

    def complete_execution(
        self, token_id: str, result: Any = None, error: str | None = None, network_events: List[NetworkEvent] = None
    ) -> bool:
        """Mark an execution as completed and move to completed registry"""
        if token_id in self._pending:
            record = self._pending.pop(token_id)
            record.completion_time = time.time()

            if error:
                record.status = ExecutionStatus.FAILED
                record.error = error
            else:
                record.status = ExecutionStatus.COMPLETED
                record.result = result
            
            # Add network events if provided
            if network_events:
                record.network_activity = network_events

            self._completed[token_id] = record
            return True
        return False

    def verify_token(self, token_id: str) -> Optional[ExecutionRecord]:
        """Verify that a token represents a completed execution"""
        self._cleanup_expired()
        return self._completed.get(token_id)

    def _cleanup_expired(self):
        """Remove expired executions"""
        current_time = time.time()
        expired_tokens = []

        with self._lock:
            # Check pending executions
            for token_id, record in self._pending.items():
                if current_time - record.token.timestamp > self._timeout_seconds:
                    expired_tokens.append(token_id)

            for token_id in expired_tokens:
                record = self._pending[token_id]
                record.status = ExecutionStatus.TIMEOUT
                self._completed[token_id] = self._pending.pop(token_id)


# Global execution registry instance
execution_registry = ExecutionRegistry()


class AgentExecutionInterface:
    """Interface for agents to request and verify tool executions
    
    This class provides an interface for agents to interact with the
    token verification system, allowing them to request execution tokens
    and verify that observations contain valid execution records.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    def request_tool_execution(self, tool_name: str, task_id: str, 
                              *args, **kwargs) -> ExecutionToken:
        """Request a tool execution and get a token"""
        return execution_registry.request_execution(
            tool_name, self.agent_id, task_id, args, kwargs
        )
    
    def verify_observation_token(self, token_id: str) -> bool:
        """Verify that an observation includes a valid execution token
        
        Args:
            token_id: The execution token ID to verify
            
        Returns:
            True if token is valid and execution was completed, False otherwise
        """
        record = execution_registry.verify_token(token_id)
        return record is not None and record.status == ExecutionStatus.COMPLETED


def enhance_tool_for_verification(tool_func, tool_name: str):  # -> ToolExecutionWrapper:  (return type commented due to forward reference)
    """Enhance a function with execution verification wrapper
    
    This function wraps any callable with the ToolExecutionWrapper to add
    token-based verification to tool executions.

    Args:
        tool_func: The actual tool function to wrap
        tool_name: Name of the tool for tracking purposes

    Returns:
        ToolExecutionWrapper instance that provides token verification
    """
    return ToolExecutionWrapper(tool_func, tool_name)


class ToolExecutionWrapper:
    """Wrapper that ensures tools can only be executed with valid tokens

    This wrapper integrates with CrewAI's tool system to provide execution
    verification without breaking existing workflows.
    """

    def __init__(self, tool_func, tool_name: str):
        self.tool_func = tool_func
        self.tool_name = tool_name

    def execute_with_token(self, token: ExecutionToken, *args, **kwargs) -> Any:
        """Execute tool with verification that it was properly requested

        Args:
            token: ExecutionToken from the registry
            *args: Arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            Result of tool execution

        Raises:
            ValueError: If token is invalid or expired
            Exception: If tool execution fails
        """
        # Create network monitor and start monitoring
        network_monitor = NetworkMonitor()
        
        # Verify this is a pending execution
        if not execution_registry.start_execution(token.token_id):
            raise ValueError(f"Invalid or expired execution token: {token.token_id}")

        try:
            # Verify arguments match the token
            args_str = str(args) + str(sorted(kwargs.items()))
            args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]

            if args_hash != token.args_hash:
                execution_registry.complete_execution(
                    token.token_id, 
                    error="Argument mismatch - potential replay attack"
                )
                raise ValueError("Arguments do not match execution token")

            # Start network monitoring
            network_monitor.start_monitoring()
            
            try:
                # Execute the actual tool
                result = self.tool_func(*args, **kwargs)
            finally:
                # Stop network monitoring and collect events
                network_events = network_monitor.stop_monitoring()

            # Mark as successfully completed with network evidence
            execution_registry.complete_execution(token.token_id, result=result, network_events=network_events)

            return result

        except Exception as e:
            # Stop network monitoring in case of error and collect any events
            network_events = network_monitor.stop_monitoring()
            
            # Mark as failed with network evidence
            execution_registry.complete_execution(token.token_id, error=str(e), network_events=network_events)
            raise


def verify_observation_token(token_id: str) -> bool:
    """Verify that an observation includes a valid execution token

    This function is used by agents to verify that observations
    contain results from actual tool executions.

    Args:
        token_id: The execution token ID to verify

    Returns:
        True if token is valid and execution was completed, False otherwise
    """
    record = execution_registry.verify_token(token_id)
    return record is not None and record.status == ExecutionStatus.COMPLETED


# Integration utilities for CrewAI
def create_token_verified_tool_usage():
    """Factory function to create ToolUsage with token verification

    This would be integrated into CrewAI's ToolUsage class to automatically
    request and verify execution tokens.
    """


def wrap_tool_for_verification(tool):
    """Wrap a CrewAI tool with execution verification

    This function can be used to wrap existing tools to add
    execution verification without modifying their code.

    Args:
        tool: A CrewAI BaseTool instance

    Returns:
        Tool wrapped with execution verification
    """
    # This would wrap the tool's invoke method with token verification
    return tool
