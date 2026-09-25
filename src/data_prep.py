"""
Football Player Wage Intelligence
Data preparation utilities.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_POSITION_MAP = {"Midfilder": "Midfielder"}

IDENTIFIER_COLUMNS = ["Player_ID"]


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load, clean, and feature-engineer the supplied football wage dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "Wage",
        "Age",
        "Club",
        "League",
        "Nation",
        "Position",
        "Apps",
        "Caps",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Wage"] = (
        df["Wage"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Fix the spelling present in the supplied source data.
    df["Position"] = df["Position"].replace(SOURCE_POSITION_MAP)

    # Remove exact duplicates first.
    df = df.drop_duplicates().reset_index(drop=True)

    # The supplied dataset does not contain player names.
    # Create a stable non-personal identifier for analytics and reporting.
    df.insert(
        0,
        "Player_ID",
        [f"P{idx:05d}" for idx in range(1, len(df) + 1)],
    )

    return add_engineered_features(df)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable ratios from existing columns."""
    out = df.copy()

    apps = out["Apps"].astype(float)
    caps = out["Caps"].astype(float)

    out["Caps_per_App"] = np.divide(
        caps,
        apps,
        out=np.zeros(len(out), dtype=float),
        where=apps.to_numpy() != 0,
    )

    # A simple career-exposure proxy; this is an analytical proxy, not a
    # measure of actual professional experience.
    career_years = np.maximum(out["Age"].astype(float) - 16.0, 1.0)

    out["Apps_per_Year"] = apps / career_years
    out["International_Exposure"] = caps / career_years

    return out


def split_features_target(df: pd.DataFrame):
    """Return model features and target, excluding identifiers."""
    x = df.drop(
        columns=["Wage"] + IDENTIFIER_COLUMNS,
        errors="ignore",
    )
    y = df["Wage"]
    return x, y
