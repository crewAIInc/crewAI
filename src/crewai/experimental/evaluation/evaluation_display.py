from collections import defaultdict
from typing import Dict, Any, List
from rich.table import Table
from rich.box import HEAVY_EDGE, ROUNDED
from collections.abc import Sequence
from crewai.experimental.evaluation.base_evaluator import AgentAggregatedEvaluationResult, AggregationStrategy, AgentEvaluationResult, MetricCategory
from crewai.experimental.evaluation import EvaluationScore
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
from crewai.utilities.llm_utils import create_llm

class EvaluationDisplayFormatter:
    def __init__(self):
        self.console_formatter = ConsoleFormatter()

    def display_evaluation_with_feedback(self, iterations_results: Dict[int, Dict[str, List[Any]]]):
        if not iterations_results:
            self.console_formatter.print("[yellow]No evaluation results to display[/yellow]")
            return

        all_agent_roles: set[str] = set()
        for iter_results in iterations_results.values():
            all_agent_roles.update(iter_results.keys())

        for agent_role in sorted(all_agent_roles):
            self.console_formatter.print(f"\n[bold cyan]Agent: {agent_role}[/bold cyan]")

            for iter_num, results in sorted(iterations_results.items()):
                if agent_role not in results or not results[agent_role]:
                    continue

                agent_results = results[agent_role]
                agent_id = agent_results[0].agent_id

                aggregated_result = self._aggregate_agent_results(
                    agent_id=agent_id,
                    agent_role=agent_role,
                    results=agent_results,
                )

                self.console_formatter.print(f"\n[bold]Iteration {iter_num}[/bold]")

                table = Table(box=ROUNDED)
                table.add_column("Metric", style="cyan")
                table.add_column("Score (1-10)", justify="center")
                table.add_column("Feedback", style="green")

                if aggregated_result.metrics:
                    for metric, evaluation_score in aggregated_result.metrics.items():
                        score = evaluation_score.score

                        if isinstance(score, (int, float)):
                            if score >= 8.0:
                                score_text = f"[green]{score:.1f}[/green]"
                            elif score >= 6.0:
                                score_text = f"[cyan]{score:.1f}[/cyan]"
                            elif score >= 4.0:
                                score_text = f"[yellow]{score:.1f}[/yellow]"
                            else:
                                score_text = f"[red]{score:.1f}[/red]"
                        else:
                            score_text = "[dim]N/A[/dim]"

                        table.add_section()
                        table.add_row(
                            metric.title(),
                            score_text,
                            evaluation_score.feedback or ""
                        )

                if aggregated_result.overall_score is not None:
                    overall_score = aggregated_result.overall_score
                    if overall_score >= 8.0:
                        overall_color = "green"
                    elif overall_score >= 6.0:
                        overall_color = "cyan"
                    elif overall_score >= 4.0:
                        overall_color = "yellow"
                    else:
                        overall_color = "red"

                    table.add_section()
                    table.add_row(
                        "Overall Score",
                        f"[{overall_color}]{overall_score:.1f}[/]",
                        "Overall agent evaluation score"
                    )

                self.console_formatter.print(table)

    def display_summary_results(self, iterations_results: Dict[int, Dict[str, List[AgentAggregatedEvaluationResult]]]):
        if not iterations_results:
            self.console_formatter.print("[yellow]No evaluation results to display[/yellow]")
            return

        self.console_formatter.print("\n")

        table = Table(title="Agent Performance Scores \n (1-10 Higher is better)", box=HEAVY_EDGE)

        table.add_column("Agent/Metric", style="cyan")

        for iter_num in sorted(iterations_results.keys()):
            run_label = f"Run {iter_num}"
            table.add_column(run_label, justify="center")

        table.add_column("Avg. Total", justify="center")

        all_agent_roles: set[str] = set()
        for results in iterations_results.values():
            all_agent_roles.update(results.keys())

        for agent_role in sorted(all_agent_roles):
            agent_scores_by_iteration = {}
            agent_metrics_by_iteration = {}

            for iter_num, results in sorted(iterations_results.items()):
                if agent_role not in results or not results[agent_role]:
                    continue

                agent_results = results[agent_role]
                agent_id = agent_results[0].agent_id

                aggregated_result = self._aggregate_agent_results(
                    agent_id=agent_id,
                    agent_role=agent_role,
                    results=agent_results,
                    strategy=AggregationStrategy.SIMPLE_AVERAGE
                )

                valid_scores = [score.score for score in aggregated_result.metrics.values()
                               if score.score is not None]
                if valid_scores:
                    avg_score = sum(valid_scores) / len(valid_scores)
                    agent_scores_by_iteration[iter_num] = avg_score

                agent_metrics_by_iteration[iter_num] = aggregated_result.metrics

            if not agent_scores_by_iteration:
                continue

            avg_across_iterations = sum(agent_scores_by_iteration.values()) / len(agent_scores_by_iteration)

            row = [f"[bold]{agent_role}[/bold]"]

            for iter_num in sorted(iterations_results.keys()):
                if iter_num in agent_scores_by_iteration:
                    score = agent_scores_by_iteration[iter_num]
                    if score >= 8.0:
                        color = "green"
                    elif score >= 6.0:
                        color = "cyan"
                    elif score >= 4.0:
                        color = "yellow"
                    else:
                        color = "red"
                    row.append(f"[bold {color}]{score:.1f}[/]")
                else:
                    row.append("-")

            if avg_across_iterations >= 8.0:
                color = "green"
            elif avg_across_iterations >= 6.0:
                color = "cyan"
            elif avg_across_iterations >= 4.0:
                color = "yellow"
            else:
                color = "red"
            row.append(f"[bold {color}]{avg_across_iterations:.1f}[/]")

            table.add_row(*row)

            all_metrics: set[Any] = set()
            for metrics in agent_metrics_by_iteration.values():
                all_metrics.update(metrics.keys())

            for metric in sorted(all_metrics, key=lambda x: x.value):
                metric_scores = []

                row = [f"  - {metric.title()}"]

                for iter_num in sorted(iterations_results.keys()):
                    if (iter_num in agent_metrics_by_iteration and
                            metric in agent_metrics_by_iteration[iter_num]):
                        metric_score = agent_metrics_by_iteration[iter_num][metric].score
                        if metric_score is not None:
                            metric_scores.append(metric_score)
                            if metric_score >= 8.0:
                                color = "green"
                            elif metric_score >= 6.0:
                                color = "cyan"
                            elif metric_score >= 4.0:
                                color = "yellow"
                            else:
                                color = "red"
                            row.append(f"[{color}]{metric_score:.1f}[/]")
                        else:
                            row.append("[dim]N/A[/dim]")
                    else:
                        row.append("-")

                if metric_scores:
                    avg = sum(metric_scores) / len(metric_scores)
                    if avg >= 8.0:
                        color = "green"
                    elif avg >= 6.0:
                        color = "cyan"
                    elif avg >= 4.0:
                        color = "yellow"
                    else:
                        color = "red"
                    row.append(f"[{color}]{avg:.1f}[/]")
                else:
                    row.append("-")

                table.add_row(*row)

            table.add_row(*[""] * (len(sorted(iterations_results.keys())) + 2))

        self.console_formatter.print(table)
        self.console_formatter.print("\n")

    def _aggregate_agent_results(
        self,
        agent_id: str,
        agent_role: str,
        results: Sequence[AgentEvaluationResult],
        strategy: AggregationStrategy = AggregationStrategy.SIMPLE_AVERAGE,
    ) -> AgentAggregatedEvaluationResult:
        metrics_by_category: dict[MetricCategory, list[EvaluationScore]] = defaultdict(list)

        for result in results:
            for metric_name, evaluation_score in result.metrics.items():
                metrics_by_category[metric_name].append(evaluation_score)

        aggregated_metrics: dict[MetricCategory, EvaluationScore] = {}
        for category, scores in metrics_by_category.items():
            valid_scores = [s.score for s in scores if s.score is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

            feedbacks = [s.feedback for s in scores if s.feedback]

            feedback_summary = None
            if feedbacks:
                if len(feedbacks) > 1:
                    feedback_summary = self._summarize_feedbacks(
                        agent_role=agent_role,
                        metric=category.title(),
                        feedbacks=feedbacks,
                        scores=[s.score for s in scores],
                        strategy=strategy
                    )
                else:
                    feedback_summary = feedbacks[0]

            aggregated_metrics[category] = EvaluationScore(
                score=avg_score,
                feedback=feedback_summary
            )

        overall_score = None
        if aggregated_metrics:
            valid_scores = [m.score for m in aggregated_metrics.values() if m.score is not None]
            if valid_scores:
                overall_score = sum(valid_scores) / len(valid_scores)

        return AgentAggregatedEvaluationResult(
            agent_id=agent_id,
            agent_role=agent_role,
            metrics=aggregated_metrics,
            overall_score=overall_score,
            task_count=len(results),
            aggregation_strategy=strategy
        )

    def _summarize_feedbacks(
        self,
        agent_role: str,
        metric: str,
        feedbacks: List[str],
        scores: List[float | None],
        strategy: AggregationStrategy
    ) -> str:
        if len(feedbacks) <= 2 and all(len(fb) < 200 for fb in feedbacks):
            return "\n\n".join([f"Feedback {i+1}: {fb}" for i, fb in enumerate(feedbacks)])

        try:
            llm = create_llm()

            formatted_feedbacks = []
            for i, (feedback, score) in enumerate(zip(feedbacks, scores)):
                if len(feedback) > 500:
                    feedback = feedback[:500] + "..."
                score_text = f"{score:.1f}" if score is not None else "N/A"
                formatted_feedbacks.append(f"Feedback #{i+1} (Score: {score_text}):\n{feedback}")

            all_feedbacks = "\n\n" + "\n\n---\n\n".join(formatted_feedbacks)

            strategy_guidance = ""
            if strategy == AggregationStrategy.BEST_PERFORMANCE:
                strategy_guidance = "Focus on the highest-scoring aspects and strengths demonstrated."
            elif strategy == AggregationStrategy.WORST_PERFORMANCE:
                strategy_guidance = "Focus on areas that need improvement and common issues across tasks."
            else:
                strategy_guidance = "Provide a balanced analysis of strengths and weaknesses across all tasks."

            prompt = [
                {"role": "system", "content": f"""You are an expert evaluator creating a comprehensive summary of agent performance feedback.
                Your job is to synthesize multiple feedback points about the same metric across different tasks.

                Create a concise, insightful summary that captures the key patterns and themes from all feedback.
                {strategy_guidance}

                Your summary should be:
                1. Specific and concrete (not vague or general)
                2. Focused on actionable insights
                3. Highlighting patterns across tasks
                4. 150-250 words in length

                The summary should be directly usable as final feedback for the agent's performance on this metric."""},
                {"role": "user", "content": f"""I need a synthesized summary of the following feedback for:

                Agent Role: {agent_role}
                Metric: {metric.title()}

                {all_feedbacks}
                """}
            ]
            assert llm is not None
            response = llm.call(prompt)

            return response

        except Exception:
            return "Synthesized from multiple tasks: " + "\n\n".join([f"- {fb[:500]}..." for fb in feedbacks])
