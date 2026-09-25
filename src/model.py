"""
Football Player Wage Intelligence
Modeling utilities.

Workflow:
1. Split data into train/test.
2. Use 5-fold cross-validation ONLY on training data for model selection.
3. Select the model with the lowest mean CV RMSE.
4. Refit the selected model on the complete training set.
5. Evaluate once on the untouched test set.

The dataset supplied for this project contains no real player-name field.
A synthetic Player_ID is used only to identify records in outputs and dashboards.
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_COLUMNS = [
    "Club",
    "League",
    "Nation",
    "Position",
]

IDENTIFIER_COLUMNS = [
    "Player_ID",
]


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for the actual model features."""

    categorical = [
        col for col in CATEGORICAL_COLUMNS
        if col in x.columns
    ]

    numerical = [
        col for col in x.columns
        if col not in categorical
    ]

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_models(x: pd.DataFrame) -> dict[str, Pipeline]:
    """Create candidate models."""

    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(x)),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(x)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=42,
                        n_jobs=-1,
                        min_samples_leaf=2,
                        max_features="sqrt",
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(x)),
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=42,
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.04,
                        loss="huber",
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep identifiers for reporting, but never use them as ML features.
    identifiers = df[
        [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    ].copy()

    X = df.drop(
        columns=["Wage"] + IDENTIFIER_COLUMNS,
        errors="ignore",
    )
    y = df["Wage"]

    # 80/20 hold-out split.
    (
        X_train,
        X_test,
        y_train,
        y_test,
        id_train,
        id_test,
    ) = train_test_split(
        X,
        y,
        identifiers,
        test_size=0.20,
        random_state=42,
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    candidate_models = make_models(X_train)
    cv_results = {}

    # Model selection happens using only training data.
    for name, model in candidate_models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
            return_train_score=False,
        )

        cv_results[name] = {
            "Model": name,
            "CV_RMSE_Mean": float(-scores["test_rmse"].mean()),
            "CV_RMSE_Std": float(scores["test_rmse"].std()),
            "CV_MAE_Mean": float(-scores["test_mae"].mean()),
            "CV_R2_Mean": float(scores["test_r2"].mean()),
        }

    cv_results_df = pd.DataFrame(
        list(cv_results.values())
    ).sort_values(
        ["CV_RMSE_Mean", "CV_MAE_Mean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best_model_name = cv_results_df.iloc[0]["Model"]
    best_model = candidate_models[best_model_name]

    # Fit on complete training data after CV selection.
    best_model.fit(X_train, y_train)

    # Median baseline.
    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train, y_train)
    baseline_pred = np.maximum(
        baseline.predict(X_test),
        0.0,
    )
    baseline_metrics = regression_metrics(
        y_test,
        baseline_pred,
    )

    # Single final evaluation on untouched test data.
    best_pred = np.maximum(
        best_model.predict(X_test),
        0.0,
    )
    best_metrics = regression_metrics(
        y_test,
        best_pred,
    )

    final_metrics = pd.DataFrame(
        [
            {
                "Model": "Median Baseline",
                **baseline_metrics,
            },
            {
                "Model": best_model_name,
                **best_metrics,
            },
        ]
    )
    final_metrics.to_csv(
        output_dir / "model_metrics.csv",
        index=False,
    )

    cv_results_df.to_csv(
        output_dir / "cv_model_comparison.csv",
        index=False,
    )

    # Build player/record-level prediction output.
    test_predictions = id_test.reset_index(drop=True).copy()
    test_features = X_test.reset_index(drop=True).copy()

    for column in test_features.columns:
        test_predictions[column] = test_features[column]

    test_predictions["Actual_Wage"] = (
        y_test.reset_index(drop=True)
    )
    test_predictions["Predicted_Wage"] = best_pred

    test_predictions["Absolute_Error"] = np.abs(
        test_predictions["Actual_Wage"]
        - test_predictions["Predicted_Wage"]
    )

    test_predictions["Wage_Difference"] = (
        test_predictions["Actual_Wage"]
        - test_predictions["Predicted_Wage"]
    )

    test_predictions["Expected_Wage_Ratio"] = np.divide(
        test_predictions["Actual_Wage"],
        test_predictions["Predicted_Wage"],
        out=np.full(
            len(test_predictions),
            np.nan,
            dtype=float,
        ),
        where=test_predictions["Predicted_Wage"] != 0,
    )

    test_predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    # Permutation importance over original, human-readable model features.
    baseline_mae = mean_absolute_error(
        y_test,
        best_pred,
    )

    rng = np.random.default_rng(42)
    permutation_rows = []

    for feature in X_test.columns:
        original_values = X_test[feature].to_numpy(copy=True)
        deltas = []

        for _ in range(10):
            shuffled = original_values.copy()
            rng.shuffle(shuffled)

            X_permuted = X_test.copy()
            X_permuted[feature] = shuffled

            perm_pred = np.maximum(
                best_model.predict(X_permuted),
                0.0,
            )

            perm_mae = mean_absolute_error(
                y_test,
                perm_pred,
            )

            deltas.append(
                perm_mae - baseline_mae
            )

        permutation_rows.append(
            {
                "Feature": feature,
                "Importance_MAE_Delta": float(np.mean(deltas)),
                "Std": float(np.std(deltas)),
            }
        )

    importance_df = pd.DataFrame(
        permutation_rows
    ).sort_values(
        "Importance_MAE_Delta",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.to_csv(
        output_dir / "permutation_importance.csv",
        index=False,
    )

    joblib.dump(
        best_model,
        output_dir / "best_model.joblib",
    )

    metadata = {
        "best_model": best_model_name,
        "selection_method": (
            "5-fold cross-validation on training data using mean RMSE"
        ),
        "final_evaluation": (
            "single evaluation on untouched 20% test set"
        ),
        "test_size": 0.20,
        "random_state": 42,
        "cv_folds": 5,
        "target": "Wage",
        "identifier_columns": IDENTIFIER_COLUMNS,
        "note": (
            "The source dataset has no player-name field. "
            "Player_ID is synthetic and used only for record identification."
        ),
        "categorical_features": CATEGORICAL_COLUMNS,
        "engineered_features": [
            "Caps_per_App",
            "Apps_per_Year",
            "International_Exposure",
        ],
        "candidate_models": list(
            candidate_models.keys()
        ),
    }

    (
        output_dir / "model_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "cv_results": cv_results_df,
        "final_metrics": final_metrics,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": best_pred,
        "importance": importance_df,
    }
