"""
Tool Execution Authenticity Verification System

This module provides comprehensive verification of tool execution authenticity to detect
when agents fabricate tool results instead of actually executing tools. Based on 2024
research into LLM tool hallucination patterns.

Key Components:
- ExecutionEvidence: Collects subprocess, filesystem, timing evidence
- ToolExecutionCertificate: Issues authenticity certificates
- ToolExecutionMonitor: Real-time monitoring with evidence collection
- ExecutionAuthenticityLevel: Classification of execution authenticity

Research Foundation:
- 2024 studies show LLMs frequently fabricate tool outputs
- Common fabrication patterns: "successfully created", "file has been written"
- Real execution leaves measurable traces: subprocess spawning, file changes
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

# Make psutil optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class ExecutionAuthenticityLevel(Enum):
    """Classification levels for tool execution authenticity."""
    VERIFIED_REAL = "verified_real"
    LIKELY_REAL = "likely_real"
    UNCERTAIN = "uncertain"
    LIKELY_FAKE = "likely_fake"
    VERIFIED_FAKE = "verified_fake"


@dataclass
class ExecutionEvidence:
    """Evidence collected during tool execution for authenticity verification."""
    # Subprocess evidence
    subprocess_spawned: bool = False
    child_processes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Filesystem evidence
    filesystem_changes: bool = False
    files_created: Set[str] = field(default_factory=set)
    files_modified: Set[str] = field(default_factory=set)
    files_deleted: Set[str] = field(default_factory=set)
    
    # Timing evidence
    execution_time_ms: float = 0.0
    
    # Network evidence (future expansion)
    network_activity: bool = False
    
    # Environment evidence
    env_changes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolExecutionCertificate:
    """Certificate issued after tool execution verification."""
    tool_name: str
    authenticity_level: ExecutionAuthenticityLevel
    evidence: ExecutionEvidence
    confidence_score: float  # 0.0 to 1.0
    fabrication_indicators: List[str] = field(default_factory=list)
    verification_timestamp: float = field(default_factory=time.time)
    
    def is_authentic(self) -> bool:
        """Check if execution is considered authentic."""
        return self.authenticity_level in [
            ExecutionAuthenticityLevel.VERIFIED_REAL,
            ExecutionAuthenticityLevel.LIKELY_REAL
        ]
    
    def is_fabricated(self) -> bool:
        """Check if execution is considered fabricated."""
        return self.authenticity_level in [
            ExecutionAuthenticityLevel.LIKELY_FAKE,
            ExecutionAuthenticityLevel.VERIFIED_FAKE
        ]


class ToolExecutionMonitor:
    """Monitors tool execution for authenticity verification."""
    
    DEFAULT_FABRICATION_PATTERNS = [
        "successfully created", "file has been written", "operation completed successfully",
        "I have created", "search results show", "I found", "analysis shows",
        "data has been", "content written to", "saved to file", "executed successfully",
        "task completed", "process finished", "output generated", "results obtained"
    ]
    
    def __init__(self, strict_mode: bool = False, fabrication_patterns: Optional[List[str]] = None):
        """
        Initialize the monitor.
        
        Args:
            strict_mode: If True, raises exceptions for likely fabricated results
            fabrication_patterns: Custom patterns to detect fabrication (uses defaults if None)
        """
        self.strict_mode = strict_mode
        self.fabrication_patterns = fabrication_patterns or self.DEFAULT_FABRICATION_PATTERNS
        self.baseline_filesystem_state: Set[str] = set()
        self.baseline_processes: Set[int] = set()
        self.start_time: float = 0.0
        
        # Statistics for analysis
        self.authenticity_counts: Dict[str, int] = {
            "verified_real": 0,
            "likely_real": 0,
            "uncertain": 0,
            "likely_fake": 0,
            "verified_fake": 0
        }
    
    def start_monitoring(self, directory: str = ".") -> None:
        """Start monitoring before tool execution."""
        self.start_time = time.time()
        
        # Capture baseline filesystem state
        self.baseline_filesystem_state = set()
        try:
            # Only monitor current directory to avoid performance issues
            if os.path.exists(directory) and os.path.isdir(directory):
                for item in os.listdir(directory):
                    filepath = os.path.join(directory, item)
                    if os.path.isfile(filepath):
                        try:
                            stat = os.stat(filepath)
                            self.baseline_filesystem_state.add(f"{filepath}:{stat.st_mtime}")
                        except (OSError, PermissionError):
                            continue
        except (OSError, PermissionError) as e:
            logger.debug(f"Filesystem monitoring limited: {e}")
        
        # Capture baseline process state
        if PSUTIL_AVAILABLE:
            try:
                current_process = psutil.Process()
                self.baseline_processes = {child.pid for child in current_process.children(recursive=True)}
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Process monitoring limited: {e}")
        else:
            logger.debug("Process monitoring disabled - psutil not available")
            self.baseline_processes = set()
    
    def stop_monitoring_and_verify(self, tool_name: str, tool_result: Any, directory: str = ".") -> ToolExecutionCertificate:
        """Stop monitoring and verify execution authenticity."""
        execution_time = (time.time() - self.start_time) * 1000  # Convert to milliseconds
        
        # Collect evidence
        evidence = ExecutionEvidence(execution_time_ms=execution_time)
        
        # Check for subprocess spawning
        if PSUTIL_AVAILABLE:
            try:
                current_process = psutil.Process()
                current_children = {child.pid for child in current_process.children(recursive=True)}
                new_processes = current_children - self.baseline_processes
                
                if new_processes:
                    evidence.subprocess_spawned = True
                    for pid in new_processes:
                        try:
                            proc = psutil.Process(pid)
                            evidence.child_processes.append({
                                'pid': pid,
                                'name': proc.name(),
                                'cmdline': ' '.join(proc.cmdline()) if proc.cmdline() else '',
                                'create_time': proc.create_time()
                            })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Process verification limited: {e}")
        else:
            logger.debug("Skipping child process detection - psutil not available")
        
        # Check for filesystem changes
        try:
            current_filesystem_state = set()
            
            if os.path.exists(directory) and os.path.isdir(directory):
                for item in os.listdir(directory):
                    filepath = os.path.join(directory, item)
                    if os.path.isfile(filepath):
                        try:
                            stat = os.stat(filepath)
                            current_state = f"{filepath}:{stat.st_mtime}"
                            current_filesystem_state.add(current_state)
                            
                            # Check if this is a new or modified file
                            if current_state not in self.baseline_filesystem_state:
                                file_baseline = f"{filepath}:"
                                baseline_entries = [entry for entry in self.baseline_filesystem_state if entry.startswith(file_baseline)]
                                if not baseline_entries:
                                    evidence.files_created.add(filepath)
                                    evidence.filesystem_changes = True
                                else:
                                    evidence.files_modified.add(filepath)
                                    evidence.filesystem_changes = True
                        except (OSError, PermissionError):
                            continue
            
            # Check for deleted files
            baseline_files = {entry.split(':')[0] for entry in self.baseline_filesystem_state}
            current_files = {entry.split(':')[0] for entry in current_filesystem_state}
            deleted_files = baseline_files - current_files
            if deleted_files:
                evidence.files_deleted.update(deleted_files)
                evidence.filesystem_changes = True
                
        except (OSError, PermissionError) as e:
            logger.debug(f"Filesystem verification limited: {e}")
        
        # Analyze authenticity
        authenticity_level = self._analyze_execution_authenticity(evidence, tool_result)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(evidence, tool_result, authenticity_level)
        
        # Detect fabrication indicators
        fabrication_indicators = self._detect_fabrication_patterns(tool_result)
        
        # Create certificate
        certificate = ToolExecutionCertificate(
            tool_name=tool_name,
            authenticity_level=authenticity_level,
            evidence=evidence,
            confidence_score=confidence_score,
            fabrication_indicators=fabrication_indicators
        )
        
        # Update statistics
        self.authenticity_counts[authenticity_level.value] += 1
        
        # Strict mode enforcement
        if self.strict_mode and certificate.is_fabricated():
            raise Exception(
                f"Tool fabrication detected for '{tool_name}'. "
                f"Authenticity: {authenticity_level.value}, "
                f"Confidence: {confidence_score:.2f}, "
                f"Indicators: {fabrication_indicators}"
            )
        
        return certificate
    
    def _analyze_execution_authenticity(self, evidence: ExecutionEvidence, tool_result: Any) -> ExecutionAuthenticityLevel:
        """Analyze collected evidence to determine execution authenticity."""
        
        # Strong positive indicators
        if evidence.subprocess_spawned and evidence.filesystem_changes:
            return ExecutionAuthenticityLevel.VERIFIED_REAL
        
        if evidence.subprocess_spawned or evidence.filesystem_changes:
            return ExecutionAuthenticityLevel.LIKELY_REAL
        
        # Check for fabrication patterns in the result
        result_text = str(tool_result).lower()
        fabrication_indicators = [
            pattern for pattern in self.fabrication_patterns 
            if pattern in result_text
        ]
        
        # Strong fabrication indicators
        if len(fabrication_indicators) >= 3:
            return ExecutionAuthenticityLevel.VERIFIED_FAKE
        
        if len(fabrication_indicators) >= 1:
            return ExecutionAuthenticityLevel.LIKELY_FAKE
        
        # No clear indicators either way
        return ExecutionAuthenticityLevel.UNCERTAIN
    
    def _calculate_confidence_score(self, evidence: ExecutionEvidence, tool_result: Any, authenticity_level: ExecutionAuthenticityLevel) -> float:
        """Calculate confidence score for the authenticity assessment."""
        score = 0.5  # Base uncertainty
        
        # Positive evidence increases confidence in real execution
        if evidence.subprocess_spawned:
            score += 0.3
        if evidence.filesystem_changes:
            score += 0.2
        if evidence.execution_time_ms > 10:  # Realistic execution time
            score += 0.1
        
        # Fabrication patterns decrease confidence in real execution
        fabrication_count = len(self._detect_fabrication_patterns(tool_result))
        score -= fabrication_count * 0.15
        
        # Adjust based on authenticity level
        if authenticity_level in [ExecutionAuthenticityLevel.VERIFIED_REAL, ExecutionAuthenticityLevel.VERIFIED_FAKE]:
            score += 0.2  # Higher confidence for verified results
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _detect_fabrication_patterns(self, tool_result: Any) -> List[str]:
        """Detect common LLM fabrication patterns in tool results."""
        result_text = str(tool_result).lower()
        return [pattern for pattern in self.fabrication_patterns if pattern in result_text]


def verify_tool_execution(tool_name: str, tool_function: Callable, *args, monitor_directory: str = ".", **kwargs) -> Tuple[Any, ToolExecutionCertificate]:
    """
    Convenience function to verify a single tool execution.
    
    Args:
        tool_name: Name of the tool being executed
        tool_function: The tool function to execute
        *args: Arguments to pass to the tool function
        monitor_directory: Directory to monitor for filesystem changes (default: ".")
        **kwargs: Keyword arguments to pass to the tool function
    
    Returns:
        Tuple of (tool_result, verification_certificate)
    """
    monitor = ToolExecutionMonitor()
    monitor.start_monitoring(monitor_directory)
    
    try:
        result = tool_function(*args, **kwargs)
    except Exception as e:
        # Still verify even if execution failed
        result = f"Execution failed: {str(e)}"
        certificate = monitor.stop_monitoring_and_verify(tool_name, result, monitor_directory)
        # Attach certificate to raised exceptions for debugging
        setattr(e, 'verification_certificate', certificate)
        raise
    finally:
        certificate = monitor.stop_monitoring_and_verify(tool_name, result, monitor_directory)
    
    return result, certificate