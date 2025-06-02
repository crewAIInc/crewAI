"""Agent state management for long-running tasks with focus on progress tracking."""

from typing import Any, Dict, List, Optional, Union, Set
from pydantic import BaseModel, Field
from datetime import datetime
import json


class CriterionProgress(BaseModel):
    """Progress tracking for a single acceptance criterion."""
    criterion: str = Field(description="The acceptance criterion")
    status: str = Field(default="not_started", description="Status: not_started, in_progress, completed")
    progress_notes: str = Field(default="", description="Specific progress made towards this criterion")
    completion_percentage: int = Field(default=0, description="Estimated completion percentage (0-100)")
    remaining_work: str = Field(default="", description="What still needs to be done for this criterion")

    # Enhanced tracking
    processed_items: Set[str] = Field(default_factory=set, description="IDs or identifiers of processed items")
    total_items_expected: Optional[int] = Field(default=None, description="Total number of items expected (if known)")
    items_to_process: List[str] = Field(default_factory=list, description="Queue of specific items to process next")
    last_updated: datetime = Field(default_factory=datetime.now)


class ProgressLog(BaseModel):
    """Single log entry for progress tracking."""
    timestamp: datetime = Field(default_factory=datetime.now)
    action: str = Field(description="What action was taken")
    result: str = Field(description="Result or outcome of the action")
    items_processed: List[str] = Field(default_factory=list, description="Items processed in this action")
    criterion: Optional[str] = Field(default=None, description="Related acceptance criterion")


class AgentState(BaseModel):
    """Enhanced state management with deterministic progress tracking.

    This state helps agents maintain focus during long executions by tracking
    specific progress against each acceptance criterion with detailed logging.
    """

    # Core planning elements
    plan: List[str] = Field(
        default_factory=list,
        description="The current plan steps"
    )

    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="Concrete criteria that must be met for task completion"
    )

    # Progress tracking
    criteria_progress: Dict[str, CriterionProgress] = Field(
        default_factory=dict,
        description="Detailed progress for each acceptance criterion"
    )

    # Data storage
    scratchpad: Dict[str, Any] = Field(
        default_factory=dict,
        description="Storage for intermediate results and data"
    )

    # Simple tracking
    current_focus: str = Field(
        default="",
        description="What the agent should be focusing on right now"
    )

    next_steps: List[str] = Field(
        default_factory=list,
        description="Immediate next steps to take"
    )

    overall_progress: int = Field(
        default=0,
        description="Overall task completion percentage (0-100)"
    )

    # Enhanced tracking
    progress_logs: List[ProgressLog] = Field(
        default_factory=list,
        description="Detailed log of all progress made"
    )

    work_queue: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Queue of specific work items to process"
    )

    # Metadata tracking
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for tracking (e.g., total count expectations)"
    )

    def initialize_criteria_progress(self) -> None:
        """Initialize progress tracking for all acceptance criteria."""
        for criterion in self.acceptance_criteria:
            if criterion not in self.criteria_progress:
                self.criteria_progress[criterion] = CriterionProgress(criterion=criterion)

    def update_criterion_progress(
        self,
        criterion: str,
        status: str,
        progress_notes: str,
        completion_percentage: int,
        remaining_work: str,
        processed_items: Optional[List[str]] = None,
        items_to_process: Optional[List[str]] = None,
        total_items_expected: Optional[int] = None
    ) -> None:
        """Update progress for a specific criterion with enhanced tracking."""
        if criterion in self.criteria_progress:
            progress = self.criteria_progress[criterion]
            progress.status = status
            progress.progress_notes = progress_notes
            progress.completion_percentage = max(0, min(100, completion_percentage))
            progress.remaining_work = remaining_work
            progress.last_updated = datetime.now()

            # Update processed items
            if processed_items:
                progress.processed_items.update(processed_items)

            # Update items to process queue
            if items_to_process is not None:
                progress.items_to_process = items_to_process

            # Update total expected if provided
            if total_items_expected is not None:
                progress.total_items_expected = total_items_expected

            # Recalculate completion percentage based on actual items if possible
            if progress.total_items_expected and progress.total_items_expected > 0:
                actual_percentage = int((len(progress.processed_items) / progress.total_items_expected) * 100)
                progress.completion_percentage = actual_percentage

            # Update overall progress
            self._recalculate_overall_progress()

    def _recalculate_overall_progress(self) -> None:
        """Recalculate overall progress based on all criteria."""
        if not self.criteria_progress:
            self.overall_progress = 0
            return

        total_progress = sum(p.completion_percentage for p in self.criteria_progress.values())
        self.overall_progress = int(total_progress / len(self.criteria_progress))

    def add_to_scratchpad(self, key: str, value: Any) -> None:
        """Add or update a value in the scratchpad."""
        self.scratchpad[key] = value

        # Analyze the data for item tracking
        self._analyze_scratchpad_for_items(key, value)

    def _analyze_scratchpad_for_items(self, key: str, value: Any) -> None:
        """Analyze scratchpad data to extract trackable items."""
        # If it's a list, try to extract IDs
        if isinstance(value, list) and value:
            item_ids = []
            for item in value:
                if isinstance(item, dict):
                    # Look for common ID fields
                    for id_field in ['id', 'ID', 'uid', 'uuid', 'message_id', 'email_id']:
                        if id_field in item:
                            item_ids.append(str(item[id_field]))
                            break

            if item_ids:
                # Store metadata about this list
                self.metadata[f"{key}_ids"] = item_ids
                self.metadata[f"{key}_count"] = len(value)

    def log_progress(self, action: str, result: str, items_processed: Optional[List[str]] = None, criterion: Optional[str] = None) -> None:
        """Add a progress log entry."""
        log_entry = ProgressLog(
            action=action,
            result=result,
            items_processed=items_processed or [],
            criterion=criterion
        )
        self.progress_logs.append(log_entry)

    def add_to_work_queue(self, work_item: Dict[str, Any]) -> None:
        """Add an item to the work queue."""
        self.work_queue.append(work_item)

    def get_next_work_item(self) -> Optional[Dict[str, Any]]:
        """Get and remove the next item from the work queue."""
        if self.work_queue:
            return self.work_queue.pop(0)
        return None

    def set_focus_and_next_steps(self, focus: str, next_steps: List[str]) -> None:
        """Update current focus and next steps."""
        self.current_focus = focus
        self.next_steps = next_steps

    def get_progress_context(self) -> str:
        """Generate a focused progress update for the agent."""
        context = f"📊 PROGRESS UPDATE (Overall: {self.overall_progress}%)\n"
        context += "="*50 + "\n\n"

        # Current focus
        if self.current_focus:
            context += f"🎯 CURRENT FOCUS: {self.current_focus}\n\n"

        # Progress on each criterion with detailed tracking
        if self.criteria_progress:
            context += "📋 ACCEPTANCE CRITERIA PROGRESS:\n"
            for criterion, progress in self.criteria_progress.items():
                status_emoji = "✅" if progress.status == "completed" else "🔄" if progress.status == "in_progress" else "⏸️"
                context += f"\n{status_emoji} {criterion}\n"

                # Show detailed progress
                if progress.total_items_expected:
                    context += f"   Progress: {len(progress.processed_items)}/{progress.total_items_expected} items ({progress.completion_percentage}%)\n"
                else:
                    context += f"   Progress: {progress.completion_percentage}%"
                    if progress.processed_items:
                        context += f" - {len(progress.processed_items)} items processed"
                    context += "\n"

                if progress.progress_notes:
                    context += f"   Notes: {progress.progress_notes}\n"

                # Show next items to process
                if progress.items_to_process and progress.status != "completed":
                    next_items = progress.items_to_process[:3]  # Show next 3
                    context += f"   Next items: {', '.join(next_items)}"
                    if len(progress.items_to_process) > 3:
                        context += f" (and {len(progress.items_to_process) - 3} more)"
                    context += "\n"

                if progress.remaining_work and progress.status != "completed":
                    context += f"   Still needed: {progress.remaining_work}\n"

        # Work queue status
        if self.work_queue:
            context += f"\n📝 WORK QUEUE: {len(self.work_queue)} items pending\n"
            next_work = self.work_queue[0]
            context += f"   Next: {next_work.get('description', 'Process next item')}\n"

        # Next steps
        if self.next_steps:
            context += f"\n📍 IMMEDIATE NEXT STEPS:\n"
            for i, step in enumerate(self.next_steps, 1):
                context += f"{i}. {step}\n"

        # Available data
        if self.scratchpad:
            context += f"\n💾 AVAILABLE DATA IN SCRATCHPAD:\n"
            for key, value in self.scratchpad.items():
                if isinstance(value, list):
                    context += f"  • '{key}' - {len(value)} items"
                    if f"{key}_ids" in self.metadata:
                        context += f" (IDs tracked)"
                    context += "\n"
                elif isinstance(value, dict):
                    context += f"  • '{key}' - dictionary data\n"
                else:
                    context += f"  • '{key}'\n"

        # Recent progress logs
        if self.progress_logs:
            context += f"\n📜 RECENT ACTIVITY:\n"
            for log in self.progress_logs[-3:]:  # Show last 3 logs
                context += f"  • {log.timestamp.strftime('%H:%M:%S')} - {log.action}"
                if log.items_processed:
                    context += f" ({len(log.items_processed)} items)"
                context += "\n"

        context += "\n" + "="*50
        return context

    def analyze_scratchpad_for_criterion_progress(self, criterion: str) -> Dict[str, Any]:
        """Analyze scratchpad data to determine specific progress on a criterion."""
        analysis = {
            "relevant_data": [],
            "item_count": 0,
            "processed_ids": set(),
            "data_completeness": 0,
            "specific_gaps": []
        }

        criterion_lower = criterion.lower()

        # Look for data that relates to this criterion
        for key, value in self.scratchpad.items():
            key_lower = key.lower()

            # Check if this data is relevant to the criterion
            is_relevant = False
            for keyword in criterion_lower.split():
                if len(keyword) > 3 and keyword in key_lower:  # Skip short words
                    is_relevant = True
                    break

            if is_relevant:
                analysis["relevant_data"].append(key)

                # Count items and extract IDs
                if isinstance(value, list):
                    analysis["item_count"] += len(value)

                    # Try to extract IDs from metadata
                    if f"{key}_ids" in self.metadata:
                        analysis["processed_ids"].update(self.metadata[f"{key}_ids"])

                elif isinstance(value, dict):
                    analysis["item_count"] += 1

        # Calculate completeness based on what we know
        if analysis["item_count"] > 0:
            # Check if criterion mentions specific numbers
            import re
            number_match = re.search(r'\b(\d+)\b', criterion)
            if number_match:
                expected_count = int(number_match.group(1))
                analysis["data_completeness"] = min(100, int((analysis["item_count"] / expected_count) * 100))
                if analysis["item_count"] < expected_count:
                    analysis["specific_gaps"].append(f"Need {expected_count - analysis['item_count']} more items")
            else:
                # For criteria without specific numbers, use heuristics
                if "all" in criterion_lower or "every" in criterion_lower:
                    # For "all" criteria, we need to be more careful
                    analysis["data_completeness"] = 50 if analysis["item_count"] > 0 else 0
                    analysis["specific_gaps"].append("Verify all items are included")
                else:
                    analysis["data_completeness"] = min(100, analysis["item_count"] * 20)  # Rough estimate

        return analysis

    def generate_specific_next_steps(self, criterion: str) -> List[str]:
        """Generate specific, actionable next steps for a criterion."""
        analysis = self.analyze_scratchpad_for_criterion_progress(criterion)
        progress = self.criteria_progress.get(criterion)
        next_steps = []

        if not progress:
            return ["Initialize progress tracking for this criterion"]

        # If we have a queue of items to process
        if progress.items_to_process:
            next_item = progress.items_to_process[0]
            next_steps.append(f"Query/process item: {next_item}")
            if len(progress.items_to_process) > 1:
                next_steps.append(f"Then process {len(progress.items_to_process) - 1} remaining items")

        # If we have processed some items but not all
        elif progress.processed_items and progress.total_items_expected:
            remaining = progress.total_items_expected - len(progress.processed_items)
            if remaining > 0:
                next_steps.append(f"Process {remaining} more items to reach target of {progress.total_items_expected}")

        # If we have data but haven't accessed it
        elif analysis["relevant_data"] and not progress.processed_items:
            for data_key in analysis["relevant_data"][:2]:  # First 2 relevant keys
                next_steps.append(f"Access and process data from '{data_key}'")

        # Generic steps based on criterion keywords
        else:
            criterion_lower = criterion.lower()
            if "email" in criterion_lower:
                next_steps.append("Use email search/fetch tool to gather emails")
            elif "analyze" in criterion_lower or "summary" in criterion_lower:
                next_steps.append("Access stored data and create analysis/summary")
            else:
                next_steps.append(f"Use appropriate tools to gather data for: {criterion}")

        return next_steps

    def reset(self) -> None:
        """Reset state for a new task."""
        self.plan = []
        self.acceptance_criteria = []
        self.criteria_progress = {}
        self.scratchpad = {}
        self.current_focus = ""
        self.next_steps = []
        self.overall_progress = 0
        self.progress_logs = []
        self.work_queue = []
        self.metadata = {}