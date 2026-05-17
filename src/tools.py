"""Інструменти агента для обробки фінансових даних."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def load_expenses(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    mean, std = series.mean(), series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    return z.abs() > threshold


def regression_forecast(df: pd.DataFrame, periods: int = 3) -> dict:
    monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().reset_index()
    monthly["month_num"] = range(len(monthly))
    X = monthly[["month_num"]].values
    y = monthly["amount"].values
    model = LinearRegression()
    model.fit(X, y)
    future_x = np.arange(len(monthly), len(monthly) + periods).reshape(-1, 1)
    predictions = model.predict(future_x)
    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(model.score(X, y)),
        "predictions": [round(float(p), 2) for p in predictions],
        "historical": monthly["amount"].tolist(),
    }


def cluster_spending(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    agg = df.groupby("category")["amount"].agg(["sum", "mean", "count"]).reset_index()
    features = agg[["sum", "mean", "count"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=min(n_clusters, len(agg)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    agg["cluster"] = labels
    clusters = {}
    for cid in sorted(agg["cluster"].unique()):
        subset = agg[agg["cluster"] == cid]
        clusters[int(cid)] = {
            "categories": subset["category"].tolist(),
            "total": round(subset["sum"].sum(), 2),
        }
    return {"clusters": clusters, "labels": labels.tolist()}


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category")["amount"]
        .agg(total="sum", mean="mean", count="count")
        .round(2)
        .reset_index()
        .sort_values("total", ascending=False)
    )
