from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from crewai.experimental.evaluation.experiment.result import ExperimentResults


class ExperimentResultsDisplay:
    def __init__(self):
        self.console = Console()

    def summary(self, experiment_results: ExperimentResults):
        total = len(experiment_results.results)
        passed = sum(1 for r in experiment_results.results if r.passed)

        table = Table(title="Experiment Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Test Cases", str(total))
        table.add_row("Passed", str(passed))
        table.add_row("Failed", str(total - passed))
        table.add_row(
            "Success Rate", f"{(passed / total * 100):.1f}%" if total > 0 else "N/A"
        )

        self.console.print(table)

    def comparison_summary(self, comparison: dict[str, Any], baseline_timestamp: str):
        self.console.print(
            Panel(
                f"[bold]Comparison with baseline run from {baseline_timestamp}[/bold]",
                expand=False,
            )
        )

        table = Table(title="Results Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="white")
        table.add_column("Details", style="dim")

        improved = comparison.get("improved", [])
        if improved:
            details = ", ".join(
                [f"{test_identifier}" for test_identifier in improved[:3]]
            )
            if len(improved) > 3:
                details += f" and {len(improved) - 3} more"
            table.add_row("✅ Improved", str(len(improved)), details)
        else:
            table.add_row("✅ Improved", "0", "")

        regressed = comparison.get("regressed", [])
        if regressed:
            details = ", ".join(
                [f"{test_identifier}" for test_identifier in regressed[:3]]
            )
            if len(regressed) > 3:
                details += f" and {len(regressed) - 3} more"
            table.add_row("❌ Regressed", str(len(regressed)), details, style="red")
        else:
            table.add_row("❌ Regressed", "0", "")

        unchanged = comparison.get("unchanged", [])
        table.add_row("⏺ Unchanged", str(len(unchanged)), "")

        new_tests = comparison.get("new_tests", [])
        if new_tests:
            details = ", ".join(new_tests[:3])
            if len(new_tests) > 3:
                details += f" and {len(new_tests) - 3} more"
            table.add_row("+ New Tests", str(len(new_tests)), details)

        missing_tests = comparison.get("missing_tests", [])
        if missing_tests:
            details = ", ".join(missing_tests[:3])
            if len(missing_tests) > 3:
                details += f" and {len(missing_tests) - 3} more"
            table.add_row("- Missing Tests", str(len(missing_tests)), details)

        self.console.print(table)
