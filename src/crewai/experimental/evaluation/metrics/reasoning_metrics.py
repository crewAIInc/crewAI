"""Agent reasoning efficiency evaluators.

This module provides evaluator implementations for:
- Reasoning efficiency
- Loop detection
- Thinking-to-action ratio
"""

import logging
import re
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.experimental.evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationScore,
    MetricCategory,
)
from crewai.experimental.evaluation.json_parser import extract_json_from_llm_response
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


class ReasoningPatternType(Enum):
    EFFICIENT = "efficient"  # Good reasoning flow
    LOOP = "loop"  # Agent is stuck in a loop
    VERBOSE = "verbose"  # Agent is unnecessarily verbose
    INDECISIVE = "indecisive"  # Agent struggles to make decisions
    SCATTERED = "scattered"  # Agent jumps between topics without focus


class ReasoningEfficiencyEvaluator(BaseEvaluator):
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.REASONING_EFFICIENCY

    def evaluate(
        self,
        agent: Agent | BaseAgent,
        execution_trace: dict[str, Any],
        final_output: TaskOutput | str,
        task: Task | None = None,
    ) -> EvaluationScore:
        task_context = ""
        if task is not None:
            task_context = f"Task description: {task.description}\nExpected output: {task.expected_output}\n"

        llm_calls = execution_trace.get("llm_calls", [])

        if not llm_calls or len(llm_calls) < 2:
            return EvaluationScore(
                score=None,
                feedback="Insufficient LLM calls to evaluate reasoning efficiency.",
            )

        total_calls = len(llm_calls)
        total_tokens = sum(call.get("total_tokens", 0) for call in llm_calls)
        avg_tokens_per_call = total_tokens / total_calls if total_calls > 0 else 0
        time_intervals = []
        has_reliable_timing = True
        for i in range(1, len(llm_calls)):
            start_time = llm_calls[i - 1].get("end_time")
            end_time = llm_calls[i].get("start_time")
            if start_time and end_time and start_time != end_time:
                try:
                    interval = end_time - start_time
                    time_intervals.append(
                        interval.total_seconds()
                        if hasattr(interval, "total_seconds")
                        else 0
                    )
                except Exception:
                    has_reliable_timing = False
            else:
                has_reliable_timing = False

        loop_detected, loop_details = self._detect_loops(llm_calls)
        pattern_analysis = self._analyze_reasoning_patterns(llm_calls)

        efficiency_metrics = {
            "total_llm_calls": total_calls,
            "total_tokens": total_tokens,
            "avg_tokens_per_call": avg_tokens_per_call,
            "reasoning_pattern": pattern_analysis["primary_pattern"].value,
            "loops_detected": loop_detected,
        }

        if has_reliable_timing and time_intervals:
            efficiency_metrics["avg_time_between_calls"] = np.mean(time_intervals)

        loop_info = (
            f"Detected {len(loop_details)} potential reasoning loops."
            if loop_detected
            else "No significant reasoning loops detected."
        )

        call_samples = self._get_call_samples(llm_calls)

        final_output = (
            final_output.raw if isinstance(final_output, TaskOutput) else final_output
        )

        prompt = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing the reasoning efficiency of an AI agent's thought process.

Evaluate the agent's reasoning efficiency across these five key subcategories:

1. Focus (0-10): How well the agent stays on topic and avoids unnecessary tangents
2. Progression (0-10): How effectively the agent builds on previous thoughts rather than repeating or circling
3. Decision Quality (0-10): How decisively and appropriately the agent makes decisions
4. Conciseness (0-10): How efficiently the agent communicates without unnecessary verbosity
5. Loop Avoidance (0-10): How well the agent avoids getting stuck in repetitive thinking patterns

For each subcategory, provide a score from 0-10 where:
- 0: Completely inefficient
- 5: Moderately efficient
- 10: Highly efficient

The overall score should be a weighted average of these subcategories.

Return your evaluation as JSON with the following structure:
{
    "overall_score": float,
    "scores": {
        "focus": float,
        "progression": float,
        "decision_quality": float,
        "conciseness": float,
        "loop_avoidance": float
    },
    "feedback": string (general feedback about overall reasoning efficiency),
    "optimization_suggestions": string (concrete suggestions for improving reasoning efficiency),
    "detected_patterns": string (describe any inefficient reasoning patterns you observe)
}""",
            },
            {
                "role": "user",
                "content": f"""
Agent role: {agent.role}
{task_context}

Reasoning efficiency metrics:
- Total LLM calls: {efficiency_metrics["total_llm_calls"]}
- Average tokens per call: {efficiency_metrics["avg_tokens_per_call"]:.1f}
- Primary reasoning pattern: {efficiency_metrics["reasoning_pattern"]}
- {loop_info}
{"- Average time between calls: {:.2f} seconds".format(efficiency_metrics.get("avg_time_between_calls", 0)) if "avg_time_between_calls" in efficiency_metrics else ""}

Sample of agent reasoning flow (chronological sequence):
{call_samples}

Agent's final output:
{final_output[:500]}... (truncated)

Evaluate the reasoning efficiency of this agent based on these interaction patterns.
Identify any inefficient reasoning patterns and provide specific suggestions for optimization.
""",
            },
        ]

        if self.llm is None:
            raise ValueError("LLM must be initialized")
        response = self.llm.call(prompt)

        try:
            evaluation_data = extract_json_from_llm_response(response)

            scores = evaluation_data.get("scores", {})
            focus = scores.get("focus", 5.0)
            progression = scores.get("progression", 5.0)
            decision_quality = scores.get("decision_quality", 5.0)
            conciseness = scores.get("conciseness", 5.0)
            loop_avoidance = scores.get("loop_avoidance", 5.0)

            overall_score = evaluation_data.get(
                "overall_score", evaluation_data.get("score", 5.0)
            )
            feedback = evaluation_data.get("feedback", "No detailed feedback provided.")
            optimization_suggestions = evaluation_data.get(
                "optimization_suggestions", "No specific suggestions provided."
            )

            detailed_feedback = "Reasoning Efficiency Evaluation:\n"
            detailed_feedback += (
                f"• Focus: {focus}/10 - Staying on topic without tangents\n"
            )
            detailed_feedback += (
                f"• Progression: {progression}/10 - Building on previous thinking\n"
            )
            detailed_feedback += f"• Decision Quality: {decision_quality}/10 - Making appropriate decisions\n"
            detailed_feedback += (
                f"• Conciseness: {conciseness}/10 - Communicating efficiently\n"
            )
            detailed_feedback += f"• Loop Avoidance: {loop_avoidance}/10 - Avoiding repetitive patterns\n\n"

            detailed_feedback += f"Feedback:\n{feedback}\n\n"
            detailed_feedback += (
                f"Optimization Suggestions:\n{optimization_suggestions}"
            )

            return EvaluationScore(
                score=float(overall_score),
                feedback=detailed_feedback,
                raw_response=response,
            )
        except Exception as e:
            logging.warning(f"Failed to parse reasoning efficiency evaluation: {e}")
            return EvaluationScore(
                score=None,
                feedback=f"Failed to parse reasoning efficiency evaluation. Raw response: {response[:200]}...",
                raw_response=response,
            )

    def _detect_loops(self, llm_calls: list[dict]) -> tuple[bool, list[dict]]:
        loop_details = []

        messages = []
        for call in llm_calls:
            content = call.get("response", "")
            if isinstance(content, str):
                messages.append(content)
            elif isinstance(content, list) and len(content) > 0:
                # Handle message list format
                messages.extend(
                    msg["content"]
                    for msg in content
                    if isinstance(msg, dict) and "content" in msg
                )

        # Simple n-gram based similarity detection
        # For a more robust implementation, consider using embedding-based similarity
        for i in range(len(messages) - 2):
            for j in range(i + 1, len(messages) - 1):
                # Check for repeated patterns (simplistic approach)
                # A more sophisticated approach would use semantic similarity
                similarity = self._calculate_text_similarity(messages[i], messages[j])
                if similarity > 0.7:  # Arbitrary threshold
                    loop_details.append(
                        {
                            "first_occurrence": i,
                            "second_occurrence": j,
                            "similarity": similarity,
                            "snippet": messages[i][:100] + "...",
                        }
                    )

        return len(loop_details) > 0, loop_details

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        text1 = re.sub(r"\s+", " ", text1.lower()).strip()
        text2 = re.sub(r"\s+", " ", text2.lower()).strip()

        # Simple Jaccard similarity on word sets
        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _analyze_reasoning_patterns(self, llm_calls: list[dict]) -> dict[str, Any]:
        call_lengths = []
        response_times = []

        for call in llm_calls:
            content = call.get("response", "")
            if isinstance(content, str):
                call_lengths.append(len(content))
            elif isinstance(content, list) and len(content) > 0:
                # Handle message list format
                total_length = 0
                for msg in content:
                    if isinstance(msg, dict) and "content" in msg:
                        total_length += len(msg["content"])
                call_lengths.append(total_length)

            start_time = call.get("start_time")
            end_time = call.get("end_time")
            if start_time and end_time:
                try:
                    response_times.append(end_time - start_time)
                except Exception as e:
                    logging.debug(f"Failed to calculate response time: {e}")

        avg_length = np.mean(call_lengths) if call_lengths else 0
        std_length = np.std(call_lengths) if call_lengths else 0
        length_trend = self._calculate_trend(call_lengths)

        primary_pattern = ReasoningPatternType.EFFICIENT
        details = "Agent demonstrates efficient reasoning patterns."

        loop_score = self._calculate_loop_likelihood(call_lengths, response_times)
        if loop_score > 0.7:
            primary_pattern = ReasoningPatternType.LOOP
            details = "Agent appears to be stuck in repetitive thinking patterns."
        elif avg_length > 1000 and std_length / avg_length < 0.3:
            primary_pattern = ReasoningPatternType.VERBOSE
            details = "Agent is consistently verbose across interactions."
        elif len(llm_calls) > 10 and length_trend > 0.5:
            primary_pattern = ReasoningPatternType.INDECISIVE
            details = (
                "Agent shows signs of indecisiveness with increasing message lengths."
            )
        elif std_length / avg_length > 0.8:
            primary_pattern = ReasoningPatternType.SCATTERED
            details = "Agent shows inconsistent reasoning flow with highly variable responses."

        return {
            "primary_pattern": primary_pattern,
            "details": details,
            "metrics": {
                "avg_length": avg_length,
                "std_length": std_length,
                "length_trend": length_trend,
                "loop_score": loop_score,
            },
        }

    def _calculate_trend(self, values: Sequence[float | int]) -> float:
        if not values or len(values) < 2:
            return 0.0

        try:
            x = np.arange(len(values))
            y = np.array(values)

            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]

            # Normalize slope to -1 to 1 range
            max_possible_slope = max(values) - min(values)
            if max_possible_slope > 0:
                normalized_slope = slope / max_possible_slope
                return max(min(normalized_slope, 1.0), -1.0)
            return 0.0
        except Exception:
            return 0.0

    def _calculate_loop_likelihood(
        self, call_lengths: Sequence[float], response_times: Sequence[float]
    ) -> float:
        if not call_lengths or len(call_lengths) < 3:
            return 0.0

        indicators = []

        if len(call_lengths) >= 4:
            repeated_lengths = 0
            for i in range(len(call_lengths) - 2):
                ratio = (
                    call_lengths[i] / call_lengths[i + 2]
                    if call_lengths[i + 2] > 0
                    else 0
                )
                if 0.85 <= ratio <= 1.15:
                    repeated_lengths += 1

            length_repetition_score = repeated_lengths / (len(call_lengths) - 2)
            indicators.append(length_repetition_score)

        if response_times and len(response_times) >= 3:
            try:
                std_time = np.std(response_times)
                mean_time = np.mean(response_times)
                if mean_time > 0:
                    time_consistency = 1.0 - (float(std_time) / float(mean_time))
                    indicators.append(max(0.0, float(time_consistency - 0.3)) * 1.5)
            except Exception as e:
                logging.debug(f"Time consistency calculation failed: {e}")

        return float(np.mean(indicators)) if indicators else 0.0

    def _get_call_samples(self, llm_calls: list[dict]) -> str:
        samples = []

        if len(llm_calls) <= 6:
            sample_indices = list(range(len(llm_calls)))
        else:
            sample_indices = [
                0,
                1,
                len(llm_calls) // 2 - 1,
                len(llm_calls) // 2,
                len(llm_calls) - 2,
                len(llm_calls) - 1,
            ]

        for idx in sample_indices:
            call = llm_calls[idx]
            content = call.get("response", "")

            if isinstance(content, str):
                sample = content
            elif isinstance(content, list) and len(content) > 0:
                sample_parts = [
                    msg["content"]
                    for msg in content
                    if isinstance(msg, dict) and "content" in msg
                ]
                sample = "\n".join(sample_parts)
            else:
                sample = str(content)

            truncated = sample[:200] + "..." if len(sample) > 200 else sample
            samples.append(f"Call {idx + 1}:\n{truncated}\n")

        return "\n".join(samples)
