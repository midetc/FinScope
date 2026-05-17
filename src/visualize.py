"""Візуалізація результатів аналізу витрат."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT = Path("output")


def plot_category_comparison(df: pd.DataFrame) -> str:
    path = OUTPUT / "expense_comparison_boxplot.png"
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="category", y="amount", hue="category", legend=False)
    plt.title("Порівняння витрат за категоріями")
    plt.xlabel("Категорія")
    plt.ylabel("Сума (грн)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return str(path)


def plot_time_series(df: pd.DataFrame, anomalies: pd.Series) -> str:
    path = OUTPUT / "expense_time_series.png"
    daily = df.groupby("date")["amount"].sum().reset_index()
    plt.figure(figsize=(12, 5))
    plt.plot(daily["date"], daily["amount"], marker="o", linewidth=1.5, label="Щоденні витрати")
    anomaly_dates = df.loc[anomalies, "date"]
    anomaly_amounts = df.loc[anomalies, "amount"]
    if len(anomaly_dates) > 0:
        plt.scatter(
            anomaly_dates,
            anomaly_amounts,
            color="red",
            s=80,
            zorder=5,
            label="Аномалії",
        )
    plt.title("Динаміка витрат та виявлення аномалій")
    plt.xlabel("Дата")
    plt.ylabel("Сума (грн)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return str(path)


def plot_category_heatmap(summary: pd.DataFrame) -> str:
    path = OUTPUT / "expense_heatmap.png"
    pivot = summary.set_index("category")[["total", "mean", "count"]]
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", linewidths=0.5)
    plt.title("Теплова карта витрат за категоріями")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return str(path)
