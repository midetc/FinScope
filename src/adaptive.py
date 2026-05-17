"""Адаптація параметрів аналізу під характеристики даних."""

import pandas as pd


def adapt_parameters(df: pd.DataFrame, session_count: int) -> dict:
    n = len(df)
    cv = df["amount"].std() / df["amount"].mean() if df["amount"].mean() > 0 else 0
    n_categories = df["category"].nunique()

    z_threshold = 3.0
    if cv > 0.5:
        z_threshold = 2.5
    elif cv < 0.2:
        z_threshold = 3.5

    n_clusters = min(3, max(2, n_categories // 2))
    forecast_periods = 3 if n >= 20 else 2

    strategy = "standard"
    if session_count >= 3:
        strategy = "experienced"
    if n < 15:
        strategy = "minimal"

    return {
        "z_threshold": z_threshold,
        "n_clusters": n_clusters,
        "forecast_periods": forecast_periods,
        "strategy": strategy,
        "cv": round(cv, 3),
    }
