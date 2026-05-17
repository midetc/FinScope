"""FinScope — веб-інтерфейс (Streamlit)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import APP_NAME, ExpenseAnalysisAgent
from src.tools import load_expenses

st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")

st.title(f"📊 {APP_NAME}")
st.caption("Агент аналізу фінансових витрат · Варіант 13 · регресія + кластеризація")

with st.sidebar:
    st.header("Керування")
    source = st.radio("Джерело даних", ["Демо-файл", "Завантажити CSV"])
    uploaded = None
    if source == "Завантажити CSV":
        uploaded = st.file_uploader("CSV (date, category, amount)", type=["csv"])
    save_memory = st.checkbox("Зберегти в пам'ять агента", value=True)
    run_btn = st.button("Запустити аналіз", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Що робить агент:**")
    st.markdown(
        "1. Планує кроки аналізу\n"
        "2. Адаптує пороги\n"
        "3. Шукає аномалії (Z-Score)\n"
        "4. Прогноз (регресія)\n"
        "5. Кластери (K-Means)\n"
        "6. Графіки + пам'ять"
    )

if not run_btn:
    st.info("Оберіть дані в боковій панелі та натисніть **Запустити аналіз**.")
    st.markdown("Або в терміналі: `python main.py`")
    st.stop()

try:
    if source == "Демо-файл":
        df = load_expenses("data/expenses_sample.csv")
    elif uploaded is None:
        st.warning("Завантажте CSV-файл.")
        st.stop()
    else:
        df = pd.read_csv(uploaded)
        df["date"] = pd.to_datetime(df["date"])
except Exception as e:
    st.error(f"Помилка читання даних: {e}")
    st.stop()

agent = ExpenseAnalysisAgent()
with st.spinner("Аналіз..."):
    results = agent.analyze(df, save_memory=save_memory)

params = results["params"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Записів", results.get("total_records", 0))
col2.metric("Витрати", f"{results.get('total_spent', 0):,.0f} грн")
col3.metric("Аномалій", results.get("anomaly_count", 0))
col4.metric("Стратегія", params.get("strategy", "—"))

st.subheader("План аналізу")
for i, step in enumerate(results.get("plan", []), 1):
    st.markdown(f"{i}. {step['description']}")

st.subheader("Параметри адаптації")
st.json({"z_threshold": params["z_threshold"], "n_clusters": params["n_clusters"], "cv": params["cv"]})

if "regression" in results:
    reg = results["regression"]
    trend = "зростання" if reg["slope"] > 0 else "зниження"
    st.subheader("Регресія (прогноз)")
    c1, c2 = st.columns(2)
    c1.write(f"Тренд: **{trend}**, R² = {reg['r2']:.3f}")
    c2.write(f"Прогноз: {reg['predictions']} грн")

if "clustering" in results:
    st.subheader("Кластеризація")
    for cid, info in results["clustering"]["clusters"].items():
        st.write(f"**Кластер {cid}:** {', '.join(info['categories'])} — {info['total']} грн")

if "comparison" in results:
    c = results["comparison"]
    st.info(f"Порівняння з попереднім аналізом: {c['difference']:+.2f} грн")

if results.get("summary"):
    st.subheader("Витрати за категоріями")
    st.dataframe(pd.DataFrame(results["summary"]), use_container_width=True)

st.subheader("Графіки")
chart_paths = results.get("charts", [])
titles = ["Порівняння категорій", "Динаміка та аномалії", "Теплова карта"]
cols = st.columns(len(chart_paths) if chart_paths else 1)
for col, path, title in zip(cols, chart_paths, titles):
    with col:
        st.caption(title)
        if Path(path).exists():
            st.image(path, use_container_width=True)
