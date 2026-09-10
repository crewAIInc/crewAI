"""
LoopDetector Middleware for CrewAI

Detects when an agent is stuck in a repetitive action loop and provides
graceful exit strategies. Addresses issue #4682.

Usage:
    from crewai.utilities.loop_detector import LoopDetector
    
    detector = LoopDetector(window_size=5, similarity_threshold=0.85)
    
    # In the agent execution loop:
    if detector.is_looping(current_action):
        action = detector.suggest_exit_strategy()
"""

from collections import deque
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import time
import hashlib


@dataclass
class ActionRecord:
    """Record of a single agent action."""
    action_type: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    result_hash: Optional[str] = None

    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:8]


class LoopDetector:
    """
    Detects repetitive patterns in agent actions and suggests exit strategies.
    
    The detector maintains a sliding window of recent actions and checks for:
    1. Exact repetitions (same action repeated N times)
    2. Semantic similarity loops (similar but not identical actions)
    3. Cyclic patterns (A->B->C->A->B->C)
    
    Args:
        window_size: Number of recent actions to track (default: 10)
        similarity_threshold: How similar two actions must be to count as 
                            "the same" (0.0-1.0, default: 0.85)
        max_repetitions: How many repetitions before declaring a loop (default: 3)
        on_loop_detected: Optional callback when a loop is detected
    """
    
    def __init__(
        self,
        window_size: int = 10,
        similarity_threshold: float = 0.85,
        max_repetitions: int = 3,
        on_loop_detected: Optional[Callable] = None,
    ):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.max_repetitions = max_repetitions
        self.on_loop_detected = on_loop_detected
        
        self._history: deque[ActionRecord] = deque(maxlen=window_size)
        self._loop_count: int = 0
        self._total_actions: int = 0
        self._loop_events: List[Dict[str, Any]] = []
    
    def record_action(self, action_type: str, content: str, 
                      tool_name: Optional[str] = None,
                      result: Optional[str] = None) -> None:
        """Record an action to the history window."""
        record = ActionRecord(
            action_type=action_type,
            content=content,
            tool_name=tool_name,
            result_hash=hashlib.md5(result.encode()).hexdigest()[:8] if result else None,
        )
        self._history.append(record)
        self._total_actions += 1
    
    def is_looping(self, current_action: Optional[str] = None) -> bool:
        """
        Check if the agent is stuck in a loop.
        
        Returns True if any loop pattern is detected:
        - Exact repetition of the last action
        - High similarity between recent actions
        - Cyclic pattern in the action window
        """
        if len(self._history) < self.max_repetitions:
            return False
        
        checks = [
            self._check_exact_repetition(),
            self._check_similarity_loop(),
            self._check_cyclic_pattern(),
        ]
        
        is_loop = any(checks)
        
        if is_loop:
            self._loop_count += 1
            event = {
                "detected_at": time.time(),
                "action_count": self._total_actions,
                "pattern_type": [
                    name for name, result in zip(
                        ["exact", "similarity", "cyclic"], checks
                    ) if result
                ],
                "recent_actions": [r.content[:100] for r in self._history],
            }
            self._loop_events.append(event)
            
            if self.on_loop_detected:
                self.on_loop_detected(event)
        
        return is_loop
    
    def _check_exact_repetition(self) -> bool:
        """Check if the last N actions are identical."""
        if len(self._history) < self.max_repetitions:
            return False
        
        recent = list(self._history)[-self.max_repetitions:]
        first_hash = recent[0].content_hash()
        return all(r.content_hash() == first_hash for r in recent)
    
    def _check_similarity_loop(self) -> bool:
        """Check if recent actions are suspiciously similar."""
        if len(self._history) < self.max_repetitions:
            return False
        
        recent = list(self._history)[-self.max_repetitions:]
        similarities = []
        
        for i in range(len(recent) - 1):
            ratio = SequenceMatcher(
                None, recent[i].content, recent[i + 1].content
            ).ratio()
            similarities.append(ratio)
        
        if not similarities:
            return False
        
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity >= self.similarity_threshold
    
    def _check_cyclic_pattern(self) -> bool:
        """
        Check for cyclic patterns (e.g., A->B->C->A->B->C).
        Uses action type sequences to detect cycles of length 2-5.
        """
        if len(self._history) < 4:
            return False
        
        types = [r.action_type for r in self._history]
        
        # Check for cycles of length 2 through 5
        for cycle_len in range(2, min(6, len(types) // 2 + 1)):
            pattern = types[-cycle_len:]
            preceding = types[-(2 * cycle_len):-cycle_len]
            
            if len(preceding) == cycle_len and pattern == preceding:
                return True
        
        return False
    
    def suggest_exit_strategy(self) -> Dict[str, Any]:
        """
        Suggest a strategy to break out of the detected loop.
        
        Returns a dict with:
        - strategy: name of the suggested strategy
        - description: human-readable explanation
        - action: suggested concrete action
        """
        if not self._loop_events:
            return {"strategy": "none", "description": "No loop detected"}
        
        last_event = self._loop_events[-1]
        pattern_types = last_event.get("pattern_type", [])
        
        strategies = []
        
        if "exact" in pattern_types:
            strategies.append({
                "strategy": "force_different_action",
                "description": "Agent is repeating the exact same action. "
                             "Force a completely different action type.",
                "action": "skip_current_and_try_alternative",
            })
        
        if "similarity" in pattern_types:
            strategies.append({
                "strategy": "increase_temperature",
                "description": "Agent actions are too similar. Increase "
                             "randomness/creativity in next action selection.",
                "action": "adjust_parameters",
                "params": {"temperature_boost": 0.3},
            })
        
        if "cyclic" in pattern_types:
            strategies.append({
                "strategy": "break_cycle",
                "description": "Agent is in a predictable cycle. Inject a "
                             "reflection step or escalate to human.",
                "action": "inject_reflection_or_escalate",
            })
        
        # If multiple patterns detected, escalate
        if len(pattern_types) > 1:
            strategies.insert(0, {
                "strategy": "escalate",
                "description": "Multiple loop patterns detected simultaneously. "
                             "Consider pausing the agent and requesting human input.",
                "action": "pause_and_escalate",
            })
        
        return strategies[0] if strategies else {
            "strategy": "general_break",
            "description": "Loop detected but pattern unclear. Try a random action.",
            "action": "random_action",
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about loop detection."""
        return {
            "total_actions": self._total_actions,
            "loops_detected": self._loop_count,
            "loop_rate": self._loop_count / max(self._total_actions, 1),
            "window_size": self.window_size,
            "current_window": len(self._history),
            "recent_loop_events": self._loop_events[-5:],
        }
    
    def reset(self) -> None:
        """Reset the detector state."""
        self._history.clear()
        self._loop_count = 0
        self._total_actions = 0
        self._loop_events.clear()
