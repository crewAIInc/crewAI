import json
from typing import Any

from crewai.agent import Agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.experimental.evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationScore,
    MetricCategory,
)
from crewai.experimental.evaluation.json_parser import extract_json_from_llm_response
from crewai.task import Task


class ToolSelectionEvaluator(BaseEvaluator):
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.TOOL_SELECTION

    def evaluate(
        self,
        agent: Agent | BaseAgent,
        execution_trace: dict[str, Any],
        final_output: str,
        task: Task | None = None,
    ) -> EvaluationScore:
        task_context = ""
        if task is not None:
            task_context = f"Task description: {task.description}"

        tool_uses = execution_trace.get("tool_uses", [])
        tool_count = len(tool_uses)
        unique_tool_types = set(
            [tool.get("tool", "Unknown tool") for tool in tool_uses]
        )

        if tool_count == 0:
            if not agent.tools:
                return EvaluationScore(
                    score=None, feedback="Agent had no tools available to use."
                )
            return EvaluationScore(
                score=None, feedback="Agent had tools available but didn't use any."
            )

        available_tools_info = ""
        if agent.tools:
            for tool in agent.tools:
                available_tools_info += f"- {tool.name}: {tool.description}\n"
        else:
            available_tools_info = "No tools available"

        tool_types_summary = "Tools selected by the agent:\n"
        for tool_type in sorted(unique_tool_types):
            tool_types_summary += f"- {tool_type}\n"

        prompt = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing if an AI agent selected the most appropriate tools for a given task.

You must evaluate based on these 2 criteria:
1. Relevance (0-10): Were the tools chosen directly aligned with the task's goals?
2. Coverage (0-10): Did the agent select ALL appropriate tools from the AVAILABLE tools?

IMPORTANT:
- ONLY consider tools that are listed as available to the agent
- DO NOT suggest tools that aren't in the 'Available tools' list
- DO NOT evaluate the quality or accuracy of tool outputs/results
- DO NOT evaluate how many times each tool was used
- DO NOT evaluate how the agent used the parameters
- DO NOT evaluate whether the agent interpreted the task correctly

Focus ONLY on whether the correct CATEGORIES of tools were selected from what was available.

Return your evaluation as JSON with these fields:
- scores: {"relevance": number, "coverage": number}
- overall_score: number (average of all scores, 0-10)
- feedback: string (focused ONLY on tool selection decisions from available tools)
- improvement_suggestions: string (ONLY suggest better selection from the AVAILABLE tools list, NOT new tools)
""",
            },
            {
                "role": "user",
                "content": f"""
Agent role: {agent.role}
{task_context}

Available tools for this agent:
{available_tools_info}

{tool_types_summary}

Based ONLY on the task description and comparing the AVAILABLE tools with those that were selected (listed above), evaluate if the agent selected the appropriate tool types for this task.

IMPORTANT:
- ONLY evaluate selection from tools listed as available
- DO NOT suggest new tools that aren't in the available tools list
- DO NOT evaluate tool usage or results
""",
            },
        ]
        if self.llm is None:
            raise ValueError("LLM must be initialized")
        response = self.llm.call(prompt)

        try:
            evaluation_data = extract_json_from_llm_response(response)
            if evaluation_data is None:
                raise ValueError("Failed to extract evaluation data from LLM response")

            scores = evaluation_data.get("scores", {})
            relevance = scores.get("relevance", 5.0)
            coverage = scores.get("coverage", 5.0)
            overall_score = float(evaluation_data.get("overall_score", 5.0))

            feedback = "Tool Selection Evaluation:\n"
            feedback += f"• Relevance: {relevance}/10 - Selection of appropriate tool types for the task\n"
            feedback += (
                f"• Coverage: {coverage}/10 - Selection of all necessary tool types\n"
            )
            if "improvement_suggestions" in evaluation_data:
                feedback += f"Improvement Suggestions:\n{evaluation_data['improvement_suggestions']}"
            else:
                feedback += evaluation_data.get(
                    "feedback", "No detailed feedback available."
                )

            return EvaluationScore(
                score=overall_score, feedback=feedback, raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Error evaluating tool selection: {e}",
                raw_response=response,
            )


class ParameterExtractionEvaluator(BaseEvaluator):
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.PARAMETER_EXTRACTION

    def evaluate(
        self,
        agent: Agent | BaseAgent,
        execution_trace: dict[str, Any],
        final_output: str,
        task: Task | None = None,
    ) -> EvaluationScore:
        task_context = ""
        if task is not None:
            task_context = f"Task description: {task.description}"
        tool_uses = execution_trace.get("tool_uses", [])
        tool_count = len(tool_uses)

        if tool_count == 0:
            return EvaluationScore(
                score=None,
                feedback="No tool usage detected. Cannot evaluate parameter extraction.",
            )

        validation_errors = [
            {
                "tool": tool_use.get("tool", "Unknown tool"),
                "error": tool_use.get("result"),
                "args": tool_use.get("args", {}),
            }
            for tool_use in tool_uses
            if not tool_use.get("success", True)
            and tool_use.get("error_type") == "validation_error"
        ]

        validation_error_rate = (
            len(validation_errors) / tool_count if tool_count > 0 else 0
        )

        param_samples = []
        for i, tool_use in enumerate(tool_uses[:5]):
            tool_name = tool_use.get("tool", "Unknown tool")
            tool_args = tool_use.get("args", {})
            success = tool_use.get("success", True) and not tool_use.get("error", False)
            error_type = tool_use.get("error_type", "") if not success else ""

            is_validation_error = error_type == "validation_error"

            sample = f"Tool use #{i + 1} - {tool_name}:\n"
            sample += f"- Parameters: {json.dumps(tool_args, indent=2)}\n"
            sample += f"- Success: {'No' if not success else 'Yes'}"

            if is_validation_error:
                sample += " (PARAMETER VALIDATION ERROR)\n"
                sample += f"- Error: {tool_use.get('result', 'Unknown error')}"
            elif not success:
                sample += f" (Other error: {error_type})\n"

            param_samples.append(sample)

        validation_errors_info = ""
        if validation_errors:
            validation_errors_info = f"\nParameter validation errors detected: {len(validation_errors)} ({validation_error_rate:.1%} of tool uses)\n"
            for i, err in enumerate(validation_errors[:3]):
                tool_name = err.get("tool", "Unknown tool")
                error_msg = err.get("error", "Unknown error")
                args = err.get("args", {})
                validation_errors_info += f"\nValidation Error #{i + 1}:\n- Tool: {tool_name}\n- Args: {json.dumps(args, indent=2)}\n- Error: {error_msg}"

            if len(validation_errors) > 3:
                validation_errors_info += (
                    f"\n...and {len(validation_errors) - 3} more validation errors."
                )
        param_samples_text = "\n\n".join(param_samples)
        prompt = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing how well an AI agent extracts and formats PARAMETER VALUES for tool calls.

Your job is to evaluate ONLY whether the agent used the correct parameter VALUES, not whether the right tools were selected or how the tools were invoked.

Evaluate parameter extraction based on these criteria:
1. Accuracy (0-10): Are parameter values correctly identified from the context/task?
2. Formatting (0-10): Are values formatted correctly for each tool's requirements?
3. Completeness (0-10): Are all required parameter values provided, with no missing information?

IMPORTANT: DO NOT evaluate:
- Whether the right tool was chosen (that's the ToolSelectionEvaluator's job)
- How the tools were structurally invoked (that's the ToolInvocationEvaluator's job)
- The quality of results from tools

Focus ONLY on the PARAMETER VALUES - whether they were correctly extracted from the context, properly formatted, and complete.

Validation errors are important signals that parameter values weren't properly extracted or formatted.

Return your evaluation as JSON with these fields:
- scores: {"accuracy": number, "formatting": number, "completeness": number}
- overall_score: number (average of all scores, 0-10)
- feedback: string (focused ONLY on parameter value extraction quality)
- improvement_suggestions: string (concrete suggestions for better parameter VALUE extraction)
""",
            },
            {
                "role": "user",
                "content": f"""
Agent role: {agent.role}
{task_context}

Parameter extraction examples:
{param_samples_text}
{validation_errors_info}

Evaluate the quality of the agent's parameter extraction for this task.
""",
            },
        ]

        if self.llm is None:
            raise ValueError("LLM must be initialized")
        response = self.llm.call(prompt)

        try:
            evaluation_data = extract_json_from_llm_response(response)
            if evaluation_data is None:
                raise ValueError("Failed to extract evaluation data from LLM response")

            scores = evaluation_data.get("scores", {})
            accuracy = scores.get("accuracy", 5.0)
            formatting = scores.get("formatting", 5.0)
            completeness = scores.get("completeness", 5.0)

            overall_score = float(evaluation_data.get("overall_score", 5.0))

            feedback = "Parameter Extraction Evaluation:\n"
            feedback += f"• Accuracy: {accuracy}/10 - Correctly identifying required parameters\n"
            feedback += f"• Formatting: {formatting}/10 - Properly formatting parameters for tools\n"
            feedback += f"• Completeness: {completeness}/10 - Including all necessary information\n\n"

            if "improvement_suggestions" in evaluation_data:
                feedback += f"Improvement Suggestions:\n{evaluation_data['improvement_suggestions']}"
            else:
                feedback += evaluation_data.get(
                    "feedback", "No detailed feedback available."
                )

            return EvaluationScore(
                score=overall_score, feedback=feedback, raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Error evaluating parameter extraction: {e}",
                raw_response=response,
            )


class ToolInvocationEvaluator(BaseEvaluator):
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.TOOL_INVOCATION

    def evaluate(
        self,
        agent: Agent | BaseAgent,
        execution_trace: dict[str, Any],
        final_output: str,
        task: Task | None = None,
    ) -> EvaluationScore:
        task_context = ""
        if task is not None:
            task_context = f"Task description: {task.description}"
        tool_uses = execution_trace.get("tool_uses", [])
        tool_errors = []
        tool_count = len(tool_uses)

        if tool_count == 0:
            return EvaluationScore(
                score=None,
                feedback="No tool usage detected. Cannot evaluate tool invocation.",
            )

        for tool_use in tool_uses:
            if not tool_use.get("success", True) or tool_use.get("error", False):
                error_info = {
                    "tool": tool_use.get("tool", "Unknown tool"),
                    "error": tool_use.get("result"),
                    "error_type": tool_use.get("error_type", "unknown_error"),
                }
                tool_errors.append(error_info)

        error_rate = len(tool_errors) / tool_count if tool_count > 0 else 0

        error_types = {}
        for error in tool_errors:
            error_type = error.get("error_type", "unknown_error")
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1

        invocation_samples = []
        for i, tool_use in enumerate(tool_uses[:5]):
            tool_name = tool_use.get("tool", "Unknown tool")
            tool_args = tool_use.get("args", {})
            success = tool_use.get("success", True) and not tool_use.get("error", False)
            error_type = tool_use.get("error_type", "") if not success else ""
            error_msg = (
                tool_use.get("result", "No error") if not success else "No error"
            )

            sample = f"Tool invocation #{i + 1}:\n"
            sample += f"- Tool: {tool_name}\n"
            sample += f"- Parameters: {json.dumps(tool_args, indent=2)}\n"
            sample += f"- Success: {'No' if not success else 'Yes'}\n"
            if not success:
                sample += f"- Error type: {error_type}\n"
                sample += f"- Error: {error_msg}"
            invocation_samples.append(sample)

        error_type_summary = ""
        if error_types:
            error_type_summary = "Error type breakdown:\n"
            for error_type, count in error_types.items():
                error_type_summary += f"- {error_type}: {count} occurrences ({(count / tool_count):.1%})\n"

        invocation_samples_text = "\n\n".join(invocation_samples)
        prompt = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing how correctly an AI agent's tool invocations are STRUCTURED.

Your job is to evaluate ONLY the structural and syntactical aspects of how the agent called tools, NOT which tools were selected or what parameter values were used.

Evaluate the agent's tool invocation based on these criteria:
1. Structure (0-10): Does the tool call follow the expected syntax and format?
2. Error Handling (0-10): Does the agent handle tool errors appropriately?
3. Invocation Patterns (0-10): Are tool calls properly sequenced, batched, or managed?

Error types that indicate invocation issues:
- execution_error: The tool was called correctly but failed during execution
- usage_error: General errors in how the tool was used structurally

IMPORTANT: DO NOT evaluate:
- Whether the right tool was chosen (that's the ToolSelectionEvaluator's job)
- Whether the parameter values are correct (that's the ParameterExtractionEvaluator's job)
- The quality of results from tools

Focus ONLY on HOW tools were invoked - the structure, format, and handling of the invocation process.

Return your evaluation as JSON with these fields:
- scores: {"structure": number, "error_handling": number, "invocation_patterns": number}
- overall_score: number (average of all scores, 0-10)
- feedback: string (focused ONLY on structural aspects of tool invocation)
- improvement_suggestions: string (concrete suggestions for better structuring of tool calls)
""",
            },
            {
                "role": "user",
                "content": f"""
Agent role: {agent.role}
{task_context}

Tool invocation examples:
{invocation_samples_text}

Tool error rate: {error_rate:.2%} ({len(tool_errors)} errors out of {tool_count} invocations)
{error_type_summary}

Evaluate the quality of the agent's tool invocation structure during this task.
""",
            },
        ]

        if self.llm is None:
            raise ValueError("LLM must be initialized")
        response = self.llm.call(prompt)

        try:
            evaluation_data = extract_json_from_llm_response(response)
            if evaluation_data is None:
                raise ValueError("Failed to extract evaluation data from LLM response")
            scores = evaluation_data.get("scores", {})
            structure = scores.get("structure", 5.0)
            error_handling = scores.get("error_handling", 5.0)
            invocation_patterns = scores.get("invocation_patterns", 5.0)

            overall_score = float(evaluation_data.get("overall_score", 5.0))

            feedback = "Tool Invocation Evaluation:\n"
            feedback += (
                f"• Structure: {structure}/10 - Following proper syntax and format\n"
            )
            feedback += f"• Error Handling: {error_handling}/10 - Appropriately handling tool errors\n"
            feedback += f"• Invocation Patterns: {invocation_patterns}/10 - Proper sequencing and management of calls\n\n"

            if "improvement_suggestions" in evaluation_data:
                feedback += f"Improvement Suggestions:\n{evaluation_data['improvement_suggestions']}"
            else:
                feedback += evaluation_data.get(
                    "feedback", "No detailed feedback available."
                )

            return EvaluationScore(
                score=overall_score, feedback=feedback, raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Error evaluating tool invocation: {e}",
                raw_response=response,
            )
