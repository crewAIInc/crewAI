"""Tool usage and step efficiency evaluators.

This module provides evaluators for assessing:
- Tool usage quality and efficiency
- Step efficiency in task execution
"""

import logging
import re
from typing import Any, Dict, List

from pydantic import Field

from .base_evaluator import BaseEvaluator, EvaluationScore, MetricCategory
from .json_parser import extract_json_from_llm_response
from ..agent import Agent
from ..task import Task
from ..llm import BaseLLM, LLM


class ToolUsageEvaluator(BaseEvaluator):
    """Evaluates the quality and efficiency of an agent's tool usage during task execution."""

    @property
    def metric_category(self) -> MetricCategory:
        """Get the metric category for this evaluator."""
        return MetricCategory.TOOL_USAGE

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: str,
    ) -> EvaluationScore:
        """Evaluate the quality and efficiency of an agent's tool usage.

        Args:
            agent: The agent to evaluate
            task: The task that was executed
            execution_trace: The execution trace for the task
            final_output: The final output from the agent

        Returns:
            EvaluationScore: Evaluation score for tool usage quality and efficiency
        """
        # Extract tool usages from the execution trace with timestamps
        steps = execution_trace.get("steps", [])
        tool_uses = []

        for step in steps:
            if step.get("type") == "tool" and "output" in step:
                tool_uses.append(step)

        # Count tools used
        tool_count = len(tool_uses)

        if tool_count == 0:
            # No tools were used
            if not agent.tools:
                # If agent has no tools, can't evaluate tool usage
                return EvaluationScore(
                    score=5.0,
                    feedback="Agent had no tools available to use."
                )
            else:
                # Agent had tools but didn't use them
                return EvaluationScore(
                    score=0.0,
                    feedback="Agent had tools available but didn't use any."
                )

        # Create a detailed summary of tool usage for the evaluator
        tool_summary = []
        unique_tools_used = set()
        tool_info = []

        for i, tool_use in enumerate(tool_uses[:10]):  # Limit to first 10 for prompt size
            tool_name = tool_use.get("tool", "Unknown tool")
            tool_args = tool_use.get("args", {})
            tool_result = tool_use.get("result", "No result recorded")
            timestamp = tool_use.get("timestamp", "Unknown time")

            # Track unique tools used
            unique_tools_used.add(tool_name)

            # Extract key information for strategic analysis
            tool_info.append({
                "index": i+1,
                "tool": tool_name,
                "args": str(tool_args),
                "timestamp": str(timestamp)
            })

            # Format result for readability
            if isinstance(tool_result, str) and len(tool_result) > 200:
                tool_result = tool_result[:200] + "..."

            tool_summary.append(f"Tool use #{i+1}:\n- Tool: {tool_name}\n- Args: {tool_args}\n- Result: {tool_result}")

        tool_usage_sample = "\n\n".join(tool_summary)

        # Calculate tool usage efficiency metrics
        total_tools = len(agent.tools) if agent.tools else 0
        unique_tool_ratio = len(unique_tools_used) / total_tools if total_tools > 0 else 1.0
        repetition_rate = 1.0 - (len(unique_tools_used) / tool_count) if tool_count > 0 else 0.0

        # Extract strategic progression information
        progression_data = self._analyze_tool_sequence(tool_info, task.description)

        # Determine appropriate examples for the task domain
        task_examples = self._get_task_domain_examples(task.description, agent.tools)

        prompt = [
            {"role": "system", "content": f"""You are an expert evaluator assessing the quality and efficiency of an AI agent's tool usage during task execution.

Evaluate the agent's tool usage based on the following criteria:

1. Relevance (0-10): Did the agent use tools that were directly relevant to accomplishing the task? Were any critical tools overlooked?

2. Efficiency (0-10): Did the agent minimize unnecessary tool calls? Did they avoid redundant searches or operations?

3. Parameter Quality (0-10): Were the parameters/inputs to tools well-formed, specific, and appropriate?

4. Strategic Usage (0-10): Did the agent use tools in a logical sequence that builds toward the goal? Did later tool usage leverage information from earlier tool usage?

5. Result Utilization (0-10): Did the agent effectively incorporate the tool results into their reasoning and final output?

Strategic Usage Assessment:
- Examine how the agent's tool usage progressed over time
- Did later tool usage build on information from earlier tools?
- Was there a logical progression toward solving the task?
- Did the agent adapt their strategy based on intermediate results?
- Is there evidence of refinement in search queries or tool parameters?

{task_examples}

Analyze the information flow across tool calls and whether the agent effectively built upon previous results.

Return your evaluation as JSON with these fields:
- scores: {{relevance: number, efficiency: number, parameter_quality: number, strategic_usage: number, result_utilization: number}}
- overall_score: number (average of all scores, 0-10)
- feedback: string (detailed assessment with specific examples)
- improvement_suggestions: string (concrete suggestions for better tool usage)
"""},
            {"role": "user", "content": f"""
Agent role: {agent.role}
Task description: {task.description}

Total tool usages: {tool_count}
Available tools: {', '.join([t.name for t in agent.tools]) if agent.tools else 'None'}
Unique tools used: {len(unique_tools_used)} out of {total_tools} available
Efficiency metrics:
- Tool repetition rate: {repetition_rate:.2f} (lower is better)
- Tool coverage: {unique_tool_ratio:.2f} (higher is better)

Tool usage progression: {progression_data}

Detailed tool usage:
{tool_usage_sample}

Agent's final output:
{final_output}

Evaluate the quality and efficiency of the agent's tool usage during this task execution.
"""}
        ]

        # Get evaluation from LLM
        response = self.llm.call(prompt)

        try:
            # Parse the response using our robust JSON parser
            evaluation_data = extract_json_from_llm_response(response)

            # Extract the subcategory scores
            scores = evaluation_data.get("scores", {})
            relevance = scores.get("relevance", 5.0)
            efficiency = scores.get("efficiency", 5.0)
            parameter_quality = scores.get("parameter_quality", 5.0)
            strategic_usage = scores.get("strategic_usage", 5.0)
            result_utilization = scores.get("result_utilization", 5.0)

            # Get the overall score
            overall_score = float(evaluation_data.get("overall_score", 5.0))

            # Format detailed feedback with subcategories
            feedback = f"Tool Usage Evaluation:\n"
            feedback += f"• Relevance: {relevance}/10 - How well the agent selected appropriate tools\n"
            feedback += f"• Efficiency: {efficiency}/10 - How well the agent minimized unnecessary tool calls\n"
            feedback += f"• Parameter Quality: {parameter_quality}/10 - How well-formed and appropriate the parameters were\n"
            feedback += f"• Strategic Usage: {strategic_usage}/10 - How logically the tools were used in sequence\n"
            feedback += f"• Result Utilization: {result_utilization}/10 - How effectively tool results were incorporated\n\n"

            # Add improvement suggestions
            if "improvement_suggestions" in evaluation_data:
                feedback += f"Improvement Suggestions:\n{evaluation_data['improvement_suggestions']}"
            else:
                feedback += evaluation_data.get("feedback", "No detailed feedback available.")

            return EvaluationScore(
                score=overall_score,
                feedback=feedback,
                raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=5.0,
                feedback=f"Error evaluating tool usage: {e}. Raw response: {response}...",
                raw_response=response
            )

    def _analyze_tool_sequence(self, tool_info, task_description):
        """Analyze the strategic usage of tools.

        Args:
            tool_info: List of dictionaries with tool usage information
            task_description: The task description for context

        Returns:
            A string description of the tool usage progression
        """
        if not tool_info:
            return "No tool usage detected"

        # Simple progression analysis
        tool_names = [info["tool"] for info in tool_info]
        unique_tools = set(tool_names)
        repeated_tools = [t for t in tool_names if tool_names.count(t) > 1]

        # Check for potential information building patterns
        has_progression = False
        if len(tool_info) >= 2:
            # Look for potential query refinement in arguments
            for i in range(1, len(tool_info)):
                prev_args = tool_info[i-1]["args"].lower()
                curr_args = tool_info[i]["args"].lower()

                # Very basic heuristic - check if current args contain words from previous args
                # This could indicate refinement of earlier queries
                prev_words = set(prev_args.split())
                if any(word in curr_args for word in prev_words if len(word) > 4):
                    has_progression = True
                    break

        result = f"Used {len(unique_tools)} unique tools with {len(repeated_tools)} repeated tool calls. "
        result += "Tool usage shows evidence of progressive information building." if has_progression else "Limited evidence of progressive information building across tool calls."

        return result

    def _get_task_domain_examples(self, task_description, tools):
        """Generate task-appropriate examples of good and poor tool usage.

        Args:
            task_description: Description of the task
            tools: Available tools

        Returns:
            String with examples relevant to the task domain
        """
        # Simple heuristic to determine task domain from description
        task_desc_lower = task_description.lower()
        available_tools = [t.name for t in tools] if tools else []

        # Research-oriented task
        if any(term in task_desc_lower for term in ["research", "find", "gather", "collect", "market", "data"]):
            return """
Examples of good strategic usage for research tasks:
1. Starting with broad information gathering searches, then focusing on specific details
2. Using information from initial search to refine parameters for follow-up searches
3. Searching for different aspects of the topic to build comprehensive understanding

Examples of poor strategic usage:
1. Repeatedly using similar search queries without refining parameters
2. Not incorporating key terms or insights from previous search results
3. Failing to search for important sub-topics or aspects mentioned in the task
"""

        # Analysis-oriented task
        elif any(term in task_desc_lower for term in ["analyze", "evaluate", "assess", "compare"]):
            return """
Examples of good strategic usage for analysis tasks:
1. First gathering necessary data points before attempting analysis
2. Using tools to obtain different perspectives or datasets for comparison
3. Validating analysis with additional targeted tool queries

Examples of poor strategic usage:
1. Jumping to analysis without sufficient data gathering
2. Not using tools to verify important claims or statistics
3. Failing to gather data from multiple sources for robust analysis
"""

        # Writing/reporting task
        elif any(term in task_desc_lower for term in ["write", "report", "create", "develop", "draft"]):
            return """
Examples of good strategic usage for writing/reporting tasks:
1. Gathering comprehensive information before starting to write
2. Using tools to fact-check specific claims or data points
3. Searching for examples or templates relevant to the required output format

Examples of poor strategic usage:
1. Writing without sufficient research or data gathering
2. Not verifying key statistics or facts with appropriate tools
3. Failing to use tools to find relevant examples or best practices
"""

        # Default examples
        else:
            return """
Examples of good strategic usage:
1. Using tools in a logical progression that builds toward the task goal
2. Selecting the most appropriate tool for each specific sub-task
3. Using information from one tool to inform the use of subsequent tools

Examples of poor strategic usage:
1. Random or illogical sequence of tool usage
2. Repeatedly using the same tool with similar parameters
3. Not leveraging the full capabilities of available tools
"""

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON data from text that might contain markdown or other formatting."""
        # Same implementation as in other evaluators
        try:
            return json.loads(text)
        except:
            json_pattern = r'``[(?:json)?\s*([\s\S]*?)\s*](cci:2://file:///Users/luzk/workspace/crewAIInc/crewAI/src/crewai/agent.py:52:0-854:55)``|{[\s\S]*}'
            match = re.search(json_pattern, text)

            if match:
                try:
                    json_str = match.group(1) if match.group(1) else match.group(0)
                    return json.loads(json_str)
                except:
                    pass

            score_match = re.search(r'(?:score|rating):\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 5.0

            return {"score": score, "feedback": text}


class StepEfficiencyEvaluator(BaseEvaluator):
    """Evaluates the efficiency of agent execution steps."""

    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.STEP_EFFICIENCY

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any,
    ) -> EvaluationScore:
        """Evaluate step efficiency.

        Measures how efficiently the agent progressed through task execution steps.

        Args:
            agent: The agent that executed the task
            task: The task that was executed
            execution_trace: The execution trace
            final_output: The final output produced by the agent

        Returns:
            EvaluationScore: Step efficiency score
        """
        # Extract execution steps from trace
        steps = execution_trace.get("steps", [])
        step_count = len(steps)

        if step_count == 0:
            return EvaluationScore(
                score=5.0,
                feedback="No execution steps recorded in the trace."
            )

        # Calculate step timing statistics
        step_times = []
        for step in steps:
            start_time = step.get("start_time")
            end_time = step.get("end_time")
            if start_time and end_time:
                try:
                    duration = end_time - start_time
                    step_times.append(duration)
                except:
                    pass

        avg_step_time = sum(step_times) / len(step_times) if step_times else 0

        # Prepare step summary
        step_summary = []
        for i, step in enumerate(steps[:5]):  # Limit to first 5 for prompt size
            step_type = step.get("type", "Unknown")
            content = step.get("content", "No content")

            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."

            step_summary.append(f"Step #{i+1} ({step_type}):\n{content}")

        steps_sample = "\n\n".join(step_summary)

        prompt = [
            {"role": "system", "content": """You are an expert evaluator assessing the efficiency of an AI agent's execution steps during task completion.

Score the step efficiency on a scale from 0-10 where:
- 0: Completely inefficient with many wasted steps or circular reasoning
- 5: Moderately efficient with some redundancy or unnecessary steps
- 10: Highly efficient with direct progress toward the goal

Consider:
1. Directness: Did the agent make steady progress toward the goal?
2. Redundancy: Did the agent repeat steps unnecessarily?
3. Step quality: Were individual steps well-formed and purposeful?
4. Planning: Did the agent demonstrate good planning in its sequence of steps?
5. Adaptability: Did the agent adjust its approach when needed?

Return your evaluation as JSON with fields 'score' (number) and 'feedback' (string).
"""},
            {"role": "user", "content": f"""
Agent role: {agent.role}
Task description: {task.description}

Total execution steps: {step_count}
Average step time: {avg_step_time:.2f} seconds

Sample of execution steps:
{steps_sample}

Agent's final output:
{final_output}

Evaluate the efficiency of the agent's execution steps during this task.
"""}
        ]

        # Get evaluation from LLM
        response = self.llm.call(prompt)

        try:
            # Parse the response
            evaluation_data = self._extract_json_from_text(response)
            return EvaluationScore(
                score=float(evaluation_data.get("score", 5.0)),
                feedback=evaluation_data.get("feedback", response),
                raw_response=response
            )
        except Exception as e:
            # Fallback if parsing fails
            return EvaluationScore(
                score=5.0,
                feedback=f"Failed to parse evaluation. Raw response: {response}",
                raw_response=response
            )

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON data from text that might contain markdown or other formatting."""
        # Same implementation as in other evaluators
        try:
            return json.loads(text)
        except:
            json_pattern = r'``[(?:json)?\s*([\s\S]*?)\s*](cci:2://file:///Users/luzk/workspace/crewAIInc/crewAI/src/crewai/agent.py:52:0-854:55)``|{[\s\S]*}'
            match = re.search(json_pattern, text)

            if match:
                try:
                    json_str = match.group(1) if match.group(1) else match.group(0)
                    return json.loads(json_str)
                except:
                    pass

            score_match = re.search(r'(?:score|rating):\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 5.0

            return {"score": score, "feedback": text}