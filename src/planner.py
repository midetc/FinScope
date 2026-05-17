"""Планування кроків аналізу фінансових витрат."""

from dataclasses import dataclass


@dataclass
class AnalysisStep:
    name: str
    description: str
    tool: str


def build_plan(data_rows: int, has_history: bool) -> list[AnalysisStep]:
    steps = [
        AnalysisStep(
            "load",
            "Завантаження та перевірка даних про витрати",
            "load_expenses",
        ),
        AnalysisStep(
            "summary",
            "Підсумок витрат за категоріями",
            "category_summary",
        ),
        AnalysisStep(
            "anomalies",
            "Виявлення аномальних витрат (Z-Score)",
            "zscore",
        ),
        AnalysisStep(
            "regression",
            "Прогнозування тренду витрат (лінійна регресія)",
            "regression",
        ),
        AnalysisStep(
            "clustering",
            "Кластеризація категорій витрат (K-Means)",
            "clustering",
        ),
        AnalysisStep(
            "visualize",
            "Побудова графіків результатів",
            "visualize",
        ),
    ]
    if data_rows < 10:
        steps = [s for s in steps if s.tool not in ("regression", "clustering")]
    if has_history:
        steps.append(
            AnalysisStep(
                "compare",
                "Порівняння з попереднім аналізом (пам'ять)",
                "memory_compare",
            )
        )
    return steps
