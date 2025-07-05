"""Agent evaluator implementations for core metrics.

This module provides evaluator implementations for:
- Goal alignment
- Knowledge retrieval
- Semantic quality
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from crewai.agent import Agent
from crewai.task import Task
from crewai.llm import BaseLLM, LLM

from crewai.evaluation.base_evaluator import BaseEvaluator, EvaluationScore, MetricCategory


class GoalAlignmentEvaluator(BaseEvaluator):
    """Evaluates how well an agent's output aligns with the task goal."""

    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.GOAL_ALIGNMENT

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any,
    ) -> EvaluationScore:
        """Evaluate goal alignment.

        Measures how well the agent's output aligns with the assigned task goal.

        Args:
            agent: The agent that executed the task
            task: The task that was executed
            execution_trace: The execution trace
            final_output: The final output produced by the agent

        Returns:
            EvaluationScore: Goal alignment score
        """
        prompt = [
            {"role": "system", "content": """You are an expert evaluator assessing how well an AI agent's output aligns with its assigned task goal.

Score the agent's goal alignment on a scale from 0-10 where:
- 0: Complete misalignment, agent did not understand or attempt the task goal
- 5: Partial alignment, agent attempted the task but missed key requirements
- 10: Perfect alignment, agent fully satisfied all task requirements

Consider:
1. Did the agent correctly interpret the task goal?
2. Did the final output directly address the requirements?
3. Did the agent focus on relevant aspects of the task?
4. Did the agent provide all requested information or deliverables?

Return your evaluation as JSON with fields 'score' (number) and 'feedback' (string).
"""},
            {"role": "user", "content": f"""
Agent role: {agent.role}
Task description: {task.description}
Expected output: {task.expected_output if hasattr(task, 'expected_output') else 'Not specified'}

Agent's final output:
{final_output}

Evaluate how well the agent's output aligns with the assigned task goal.
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
        try:
            return json.loads(text)
        except:
            # Try to extract JSON from markdown code blocks
            json_pattern = r'``[(?:json)?\s*([\s\S]*?)\s*](cci:2://file:///Users/luzk/workspace/crewAIInc/crewAI/src/crewai/agent.py:52:0-854:55)``|{[\s\S]*}'
            match = re.search(json_pattern, text)

            if match:
                try:
                    json_str = match.group(1) if match.group(1) else match.group(0)
                    return json.loads(json_str)
                except:
                    pass

            # Fallback: extract score using regex
            score_match = re.search(r'(?:score|rating):\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 5.0

            return {"score": score, "feedback": text}


class KnowledgeRetrievalEvaluator(BaseEvaluator):
    """Evaluates the effectiveness of knowledge retrieval during task execution."""

    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.KNOWLEDGE_RETRIEVAL

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any,
    ) -> EvaluationScore:
        """Evaluate knowledge retrieval effectiveness.

        Measures how well the agent retrieved and used knowledge during task execution.

        Args:
            agent: The agent that executed the task
            task: The task that was executed
            execution_trace: The execution trace
            final_output: The final output produced by the agent

        Returns:
            EvaluationScore: Knowledge retrieval score
        """
        # Extract knowledge retrieval operations from trace
        retrievals = execution_trace.get("knowledge_retrievals", [])
        retrieval_count = len(retrievals)

        if retrieval_count == 0:
            # No knowledge retrievals performed - check if task required knowledge
            if not agent.knowledge:
                # Task didn't require knowledge, so this metric isn't relevant
                return EvaluationScore(
                    score=5.0,
                    feedback="Knowledge retrieval was not required for this task."
                )
            else:
                # Task had a knowledge base but agent didn't use it
                return EvaluationScore(
                    score=2.0,
                    feedback="Agent had access to a knowledge base but didn't use it."
                )

        # Prepare the knowledge retrieval evaluation
        retrieval_texts = []
        for retrieval in retrievals:
            query = retrieval.get("query", "Unknown query")
            documents = retrieval.get("documents", [])
            doc_texts = [f"- {doc[:200]}..." for doc in documents[:3]]
            retrieval_texts.append(f"Query: {query}\nRetrieved documents:\n" + "\n".join(doc_texts))

        retrieval_sample = "\n\n".join(retrieval_texts[:3])  # Limit to first 3 for prompt size

        prompt = [
            {"role": "system", "content": """You are an expert evaluator assessing the effectiveness of knowledge retrieval during an AI agent's task execution.

Score the knowledge retrieval on a scale from 0-10 where:
- 0: Completely ineffective retrieval that provided no value
- 5: Moderately effective retrieval with some relevant information
- 10: Highly effective retrieval that provided crucial information for the task

Consider:
1. Were the retrieval queries well-formed and relevant to the task?
2. Did the retrieved documents contain information relevant to the task?
3. Is there evidence the agent incorporated the retrieved knowledge into its output?
4. Was the retrieval strategy efficient (e.g., not too many redundant queries)?

Return your evaluation as JSON with fields 'score' (number) and 'feedback' (string).
"""},
            {"role": "user", "content": f"""
Agent role: {agent.role}
Task description: {task.description}

Knowledge retrieval operations performed: {retrieval_count}

Sample of knowledge retrievals:
{retrieval_sample}

Agent's final output:
{final_output}

Evaluate the effectiveness of the knowledge retrieval during this task execution.
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
        # Same implementation as in GoalAlignmentEvaluator
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


class SemanticQualityEvaluator(BaseEvaluator):
    """Evaluates the semantic quality and reasoning of the agent's output."""

    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.SEMANTIC_QUALITY

    def evaluate(
        self,
        agent: Agent,
        task: Task,
        execution_trace: Dict[str, Any],
        final_output: Any,
    ) -> EvaluationScore:
        """Evaluate semantic quality and reasoning.

        Measures the clarity, coherence, and reasoning quality of the agent's output.

        Args:
            agent: The agent that executed the task
            task: The task that was executed
            execution_trace: The execution trace
            final_output: The final output produced by the agent

        Returns:
            EvaluationScore: Semantic quality score
        """
        prompt = [
            {"role": "system", "content": """You are an expert evaluator assessing the semantic quality of an AI agent's output.

Score the semantic quality on a scale from 0-10 where:
- 0: Completely incoherent, confusing, or logically flawed output
- 5: Moderately clear and logical output with some issues
- 10: Exceptionally clear, coherent, and logically sound output

Consider:
1. Is the output well-structured and organized?
2. Is the reasoning logical and well-supported?
3. Is the language clear, precise, and appropriate for the task?
4. Are claims supported by evidence when appropriate?
5. Is the output free from contradictions and logical fallacies?

Return your evaluation as JSON with fields 'score' (number) and 'feedback' (string).
"""},
            {"role": "user", "content": f"""
Agent role: {agent.role}
Task description: {task.description}

Agent's final output:
{final_output}

Evaluate the semantic quality and reasoning of this output.
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