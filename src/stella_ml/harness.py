"""High-level orchestration harness for query/problem + data workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stella_ml.analytics import (
    EDAReport,
    apply_cleaning_operations,
    auto_eda,
    explore_chart,
    load_tabular_file,
)
from stella_ml.feasibility import (
    FeasibilityReport,
    HardwareProfile,
    autoimpute_experiment_specs,
    generate_feasibility_chain,
)


@dataclass(slots=True)
class ProblemAssessment:
    objective: str
    requires_data: bool
    automl_recommended: bool
    follow_up_question: str | None


@dataclass(slots=True)
class HarnessResult:
    assessment: ProblemAssessment
    eda_report: EDAReport | None = None
    insight_summary: list[str] | None = None
    chart_paths: list[str] | None = None


class OpenClawStyleHarness:
    """Orchestrates user requests similar to modern agent frameworks."""

    def evaluate_problem(self, user_query: str) -> ProblemAssessment:
        lower = user_query.lower()
        objective = "general_analysis"
        requires_data = any(k in lower for k in ["dataset", "csv", "excel", "table", "rows"])
        automl_recommended = any(k in lower for k in ["predict", "forecast", "classification", "regression", "automl"])

        if any(k in lower for k in ["classification", "classify"]):
            objective = "classification"
        elif any(k in lower for k in ["regression", "forecast", "predict"]):
            objective = "regression"
        elif any(k in lower for k in ["cluster", "segment"]):
            objective = "clustering"
        elif any(k in lower for k in ["eda", "explore", "insight"]):
            objective = "exploratory_analysis"

        follow_up = (
            "Would you like AutoML enabled for model search, or should I run guided statistical analysis?"
            if automl_recommended
            else None
        )

        return ProblemAssessment(
            objective=objective,
            requires_data=requires_data,
            automl_recommended=automl_recommended,
            follow_up_question=follow_up,
        )

    def isHardwareFeasible(
        self,
        hypothesis: str,
        hardware: HardwareProfile | None = None,
    ) -> list[tuple[str, FeasibilityReport]]:
        """Auto-impute experiment specs and evaluate local hardware feasibility."""
        chain = generate_feasibility_chain(hypothesis, hardware=hardware)
        return [(spec.name, report) for spec, report in chain]

    def plan_experiments(self, hypothesis: str) -> list[str]:
        """Generate candidate experiment names from hypothesis semantics."""
        return [spec.name for spec in autoimpute_experiment_specs(hypothesis)]

    def run_data_flow(
        self,
        file_path: str | Path,
        cleaning_ops: list[dict[str, object]] | None = None,
        charts: list[dict[str, str]] | None = None,
    ) -> tuple[EDAReport, list[str]]:
        rows, inferred_types = load_tabular_file(file_path)

        if cleaning_ops:
            rows = apply_cleaning_operations(rows, inferred_types, cleaning_ops)

        report = auto_eda(rows, inferred_types)
        chart_paths: list[str] = []
        for chart in charts or [{"chart_type": "auto", "output_path": "artifacts/explore_auto.png"}]:
            path = explore_chart(
                rows,
                inferred_types,
                output_path=chart.get("output_path", "artifacts/explore_auto.png"),
                chart_type=chart.get("chart_type", "auto"),
                x=chart.get("x"),
                y=chart.get("y"),
            )
            chart_paths.append(str(path))

        return report, chart_paths

    def solve(
        self,
        user_query: str,
        file_path: str | Path | None = None,
        cleaning_ops: list[dict[str, object]] | None = None,
        charts: list[dict[str, str]] | None = None,
    ) -> HarnessResult:
        assessment = self.evaluate_problem(user_query)
        if file_path is None:
            return HarnessResult(assessment=assessment)

        report, chart_paths = self.run_data_flow(file_path, cleaning_ops=cleaning_ops, charts=charts)
        return HarnessResult(
            assessment=assessment,
            eda_report=report,
            insight_summary=report.recommendations,
            chart_paths=chart_paths,
        )
