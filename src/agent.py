"""FinScope — агент аналізу фінансових витрат."""

from pathlib import Path

import numpy as np
import pandas as pd

APP_NAME = "FinScope"


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

from src.adaptive import adapt_parameters
from src.memory import AgentMemory
from src.planner import AnalysisStep, build_plan
from src.tools import (
    category_summary,
    cluster_spending,
    compute_zscore_anomalies,
    load_expenses,
    regression_forecast,
)
from src.visualize import plot_category_comparison, plot_category_heatmap, plot_time_series


class ExpenseAnalysisAgent:
    def __init__(self, data_path: str = "data/expenses_sample.csv"):
        self.data_path = data_path
        self.memory = AgentMemory()
        Path("output").mkdir(exist_ok=True)

    def analyze(self, df: pd.DataFrame | None = None, save_memory: bool = True) -> dict:
        if df is None:
            df = load_expenses(self.data_path)
        params = adapt_parameters(df, self.memory.session_count())
        plan = build_plan(len(df), self.memory.get_last_analysis() is not None)
        results = self._execute_plan(df, plan, params, quiet=True)
        charts = self._visualize(df, results, quiet=True)
        results["charts"] = charts
        results["params"] = params
        results["plan"] = [{"step": s.name, "description": s.description} for s in plan]
        del results["_anomalies"]
        if save_memory:
            self.memory.add_session([s.name for s in plan], _to_json_safe(results))
        return results

    def run(self) -> dict:
        print("=" * 60)
        print(f"  {APP_NAME} — агент аналізу фінансових витрат (Варіант 13)")
        print("=" * 60)

        df = load_expenses(self.data_path)
        params = adapt_parameters(df, self.memory.session_count())
        plan = build_plan(len(df), self.memory.get_last_analysis() is not None)

        print(f"\n[АДАПТАЦІЯ] Стратегія: {params['strategy']}, Z-поріг: {params['z_threshold']}")
        print(f"\n[ПЛАН] Кроків аналізу: {len(plan)}")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step.description}")

        results = self._execute_plan(df, plan, params)
        charts = self._visualize(df, results)
        results["charts"] = charts
        results["params"] = params
        results["plan"] = [{"step": s.name, "description": s.description} for s in plan]
        del results["_anomalies"]

        self.memory.add_session([s.name for s in plan], _to_json_safe(results))
        self._print_report(results)
        return results

    def _execute_plan(
        self, df, plan: list[AnalysisStep], params: dict, quiet: bool = False
    ) -> dict:
        results: dict = {}
        for step in plan:
            if not quiet:
                print(f"\n[→] {step.description}...")
            if step.tool == "load_expenses":
                results["total_records"] = len(df)
                results["total_spent"] = round(df["amount"].sum(), 2)
            elif step.tool == "category_summary":
                summary = category_summary(df)
                results["summary"] = summary.to_dict(orient="records")
            elif step.tool == "zscore":
                anomalies = compute_zscore_anomalies(df["amount"], params["z_threshold"])
                results["anomaly_count"] = int(anomalies.sum())
                results["anomaly_indices"] = df.index[anomalies].tolist()
            elif step.tool == "regression":
                reg = regression_forecast(df, params["forecast_periods"])
                results["regression"] = reg
            elif step.tool == "clustering":
                cl = cluster_spending(df, params["n_clusters"])
                results["clustering"] = cl
            elif step.tool == "memory_compare":
                prev = self.memory.get_last_analysis()
                if prev and "total_spent" in prev:
                    diff = results.get("total_spent", 0) - prev["total_spent"]
                    results["comparison"] = {
                        "previous_total": prev["total_spent"],
                        "difference": round(diff, 2),
                    }
        results["_anomalies"] = compute_zscore_anomalies(
            df["amount"], params["z_threshold"]
        )
        return results

    def _visualize(self, df, results: dict, quiet: bool = False) -> list[str]:
        if not quiet:
            print("\n[→] Побудова графіків...")
        charts = []
        charts.append(plot_category_comparison(df))
        charts.append(plot_time_series(df, results["_anomalies"]))
        summary_df = category_summary(df)
        charts.append(plot_category_heatmap(summary_df))
        return charts

    def _print_report(self, results: dict) -> None:
        print("\n" + "=" * 60)
        print("  РЕЗУЛЬТАТИ АНАЛІЗУ")
        print("=" * 60)
        print(f"  Записів: {results.get('total_records', '—')}")
        print(f"  Загальні витрати: {results.get('total_spent', '—')} грн")
        print(f"  Аномалій виявлено: {results.get('anomaly_count', 0)}")

        if "regression" in results:
            reg = results["regression"]
            trend = "зростання" if reg["slope"] > 0 else "зниження"
            print(f"  Тренд (регресія): {trend}, R²={reg['r2']:.3f}")
            print(f"  Прогноз на {len(reg['predictions'])} міс.: {reg['predictions']} грн")

        if "clustering" in results:
            print("  Кластери витрат:")
            for cid, info in results["clustering"]["clusters"].items():
                cats = ", ".join(info["categories"])
                print(f"    Кластер {cid}: [{cats}] — {info['total']} грн")

        if "comparison" in results:
            c = results["comparison"]
            print(f"  Порівняння з попереднім: {c['difference']:+.2f} грн")

        print("\n  Графіки збережено:")
        for ch in results.get("charts", []):
            print(f"    → {ch}")
        print("=" * 60)
