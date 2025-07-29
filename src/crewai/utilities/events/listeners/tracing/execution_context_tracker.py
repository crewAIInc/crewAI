from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ExecutionContext:
    """Represents a single execution context in the stack"""

    type: str  # "flow", "crew", "task", "agent"
    id: str
    name: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level: int = 0


class ExecutionContextTracker:
    """Single responsibility: Track execution context hierarchy"""

    def __init__(self):
        self.context_stack: List[ExecutionContext] = []

    def push_context(
        self, context_type: str, context_id: str, context_name: Optional[str] = None
    ):
        """Push new execution context onto stack"""
        context = ExecutionContext(
            type=context_type,
            id=context_id,
            name=context_name,
            level=len(self.context_stack),
        )
        self.context_stack.append(context)

    def pop_context(self, context_type: str) -> Optional[ExecutionContext]:
        """Pop context when execution completes"""
        if self.context_stack and self.context_stack[-1].type == context_type:
            return self.context_stack.pop()
        return None

    def get_current_context(
        self, context_type: Optional[str] = None
    ) -> Optional[ExecutionContext]:
        """Get current context of specific type or top context"""
        if not context_type:
            return self.context_stack[-1] if self.context_stack else None

        # Find most recent context of this type
        for context in reversed(self.context_stack):
            if context.type == context_type:
                return context
        return None

    def get_execution_path(self) -> str:
        """Get full execution path from root to current"""
        return "/".join([f"{ctx.type}:{ctx.id}" for ctx in self.context_stack])

    def get_context_correlations(self) -> Dict[str, str]:
        """Generate correlation IDs from current context stack"""
        correlations = {}

        # Add all context IDs
        for context in self.context_stack:
            correlations[f"{context.type}_id"] = context.id
            if context.name:
                correlations[f"{context.type}_name"] = context.name

        # Add execution path
        correlations["execution_path"] = self.get_execution_path()

        # Add current parent info
        current = self.get_current_context()
        if current:
            correlations["parent_type"] = current.type
            correlations["parent_id"] = current.id
            correlations["nesting_level"] = str(current.level)

        return correlations

    def get_nesting_level(self) -> int:
        """Get current nesting level"""
        return len(self.context_stack)

    def is_root_level(self) -> bool:
        """Check if we're at root level (no active contexts)"""
        return len(self.context_stack) == 0


class PrivacyFilter:
    """Single responsibility: Filter sensitive data"""

    def __init__(self, privacy_level: str = "standard"):
        self.privacy_level = privacy_level
        self.sensitive_keywords = [
            "api_key",
            "password",
            "secret",
            "token",
            "credential",
            "auth",
        ]

    def filter_content(self, content: str) -> str:
        """Apply privacy filtering based on privacy level"""
        if not content:
            return content

        if self.privacy_level == "minimal":
            return "Content filtered for privacy"
        elif self.privacy_level == "standard":
            # Basic filtering - remove obvious sensitive data
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in self.sensitive_keywords):
                return "Content filtered - contains sensitive data"
            # Truncate long content
            return content[:200] + "..." if len(content) > 200 else content
        elif self.privacy_level == "full":
            return content

        return content

    def filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter dictionary values"""
        filtered = {}
        for key, value in data.items():
            if isinstance(value, str):
                filtered[key] = self.filter_content(value)
            elif isinstance(value, dict):
                filtered[key] = self.filter_dict(value)
            else:
                filtered[key] = value
        return filtered
