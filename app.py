from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_prep import load_data

DATA_PATH = ROOT / "data" / "SalaryPrediction.csv"
CLEAN_PATH = ROOT / "data" / "cleaned_players.csv"
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "model_metrics.csv"
PRED_PATH = ROOT / "artifacts" / "test_predictions.csv"
IMPORTANCE_PATH = ROOT / "artifacts" / "permutation_importance.csv"


st.set_page_config(
    page_title="Football Player Wage Intelligence",
    layout="wide",
)

st.title("⚽ Football Player Wage Intelligence")
st.caption("Data analytics + regression modeling + explainable wage benchmarking")

@st.cache_data
def get_data():
    return load_data(DATA_PATH)

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

df = get_data()
model = get_model()

if model is None:
    st.warning("Model artifact not found. Run `python train.py` first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Player Valuation", "Model Performance", "Data Explorer"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(df):,}")
    c2.metric("Average Wage", f"${df['Wage'].mean()/1e6:.2f}M")
    c3.metric("Median Wage", f"${df['Wage'].median()/1e6:.2f}M")
    c4.metric("Max Wage", f"${df['Wage'].max()/1e6:.2f}M")

    st.subheader("Wage by Position")
    by_pos = (
        df.groupby("Position", as_index=False)["Wage"]
        .mean()
        .sort_values("Wage", ascending=False)
    )
    st.bar_chart(by_pos.set_index("Position"))

    st.subheader("Wage Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["Wage"], bins=50)
    ax.set_xlabel("Wage")
    ax.set_ylabel("Players")
    ax.set_title("Distribution of Player Wages")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Wage vs International Experience")
    scatter_df = df[["Caps", "Wage"]].copy()
    st.scatter_chart(scatter_df, x="Caps", y="Wage")

with tab2:
    st.subheader("Expected Wage vs Observed Wage")

    league = st.selectbox("League", ["All"] + sorted(df["League"].unique().tolist()))
    filtered = df.copy()
    if league != "All":
        filtered = filtered[filtered["League"] == league]

    position = st.selectbox("Position", ["All"] + sorted(df["Position"].unique().tolist()))
    if position != "All":
        filtered = filtered[filtered["Position"] == position]

    club = st.selectbox("Club", ["All"] + sorted(filtered["Club"].unique().tolist()))
    if club != "All":
        filtered = filtered[filtered["Club"] == club]

    if len(filtered) == 0:
        st.info("No players match these filters.")
    else:
        # Predict for the selected records using the final trained model.
        features = filtered.drop(columns=["Wage", "Player_ID"], errors="ignore")
        filtered = filtered.copy()
        filtered["Predicted_Wage"] = np.maximum(model.predict(features), 0.0)
        filtered["Wage_Difference"] = filtered["Wage"] - filtered["Predicted_Wage"]
        filtered["Expected_Wage_Ratio"] = np.divide(
            filtered["Wage"],
            filtered["Predicted_Wage"],
            out=np.full(len(filtered), np.nan, dtype=float),
            where=filtered["Predicted_Wage"] != 0,
        )

        display_cols = [
            "Player_ID",
            "Club", "League", "Nation", "Position",
            "Age", "Apps", "Caps", "Wage",
            "Predicted_Wage", "Wage_Difference", "Expected_Wage_Ratio"
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered.sort_values("Wage_Difference", ascending=False)[display_cols],
            use_container_width=True,
            column_config={
                "Wage": st.column_config.NumberColumn("Actual Wage", format="$%.0f"),
                "Predicted_Wage": st.column_config.NumberColumn("Estimated Wage", format="$%.0f"),
                "Wage_Difference": st.column_config.NumberColumn("Observed - Estimated", format="$%.0f"),
                "Expected_Wage_Ratio": st.column_config.NumberColumn("Observed / Estimated", format="%.2fx"),
            },
        )

        st.caption(
            "The dataset does not contain real player names, so Player_ID is a synthetic record identifier. "
            "Interpretation: the wage difference is descriptive of the model's estimate; "
            "it is not evidence that a player is objectively overpaid or underpaid."
        )

with tab3:
    st.subheader("Model Comparison")
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics, use_container_width=True)

    best_row = metrics[metrics["Model"] != "Median Baseline"].sort_values("RMSE").iloc[0]
    a, b, c = st.columns(3)
    a.metric("Selected model", best_row["Model"])
    b.metric("Test MAE", f"${best_row['MAE']:,.0f}")
    c.metric("Test R²", f"{best_row['R2']:.3f}")

    st.subheader("Permutation Importance")
    imp = pd.read_csv(IMPORTANCE_PATH).sort_values("Importance_MAE_Delta")
    st.bar_chart(imp.set_index("Feature")["Importance_MAE_Delta"])

    st.subheader("Actual vs Predicted")
    preds = pd.read_csv(PRED_PATH)
    st.scatter_chart(
        preds,
        x="Actual_Wage",
        y="Predicted_Wage",
    )

with tab4:
    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)
