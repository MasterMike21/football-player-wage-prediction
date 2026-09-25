from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SalaryPrediction.csv"
ARTIFACTS = ROOT / "artifacts"
SRC_PATH = ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from data_prep import load_data
from model import train_and_evaluate


def main():
    print("=" * 70)
    print("FOOTBALL PLAYER WAGE INTELLIGENCE")
    print("=" * 70)

    raw_df = pd.read_csv(DATA_PATH)
    raw_rows = len(raw_df)
    raw_duplicates = int(raw_df.duplicated().sum())

    print(f"\nRaw dataset rows: {raw_rows:,}")
    print(f"Exact duplicate rows: {raw_duplicates:,}")

    df = load_data(DATA_PATH)

    # Save the exact cleaned dataset consumed by modeling and Power BI.
    df.to_csv(
        ROOT / "data" / "cleaned_players.csv",
        index=False,
    )

    print(f"Rows after cleaning: {len(df):,}")
    print("\nNote: source data has no player names.")
    print("Synthetic Player_ID values were created for record-level reporting.")

    print("\nFeatures used by the ML models:")
    for column in df.columns:
        if column not in {"Player_ID", "Wage"}:
            print(f"  - {column}")

    print("\n" + "-" * 70)
    print("Running 5-fold cross-validation on training data...")

    result = train_and_evaluate(df, ARTIFACTS)

    print("\nCROSS-VALIDATION RESULTS")
    print(
        result["cv_results"].to_string(index=False)
    )

    print("\nFINAL TEST-SET RESULTS")
    print(
        result["final_metrics"].to_string(index=False)
    )

    print("\nSelected model:")
    print(f"  {result['best_model_name']}")

    print("\nIMPORTANT:")
    print("  The test set was NOT used for model selection.")
    print("  It was used only once for final evaluation.")

    metadata_path = ARTIFACTS / "model_metadata.json"
    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    metadata.update(
        {
            "raw_dataset_rows": raw_rows,
            "exact_duplicates_removed": raw_duplicates,
            "clean_dataset_rows": len(df),
        }
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nArtifacts generated:")
    print("  ✓ cleaned_players.csv")
    print("  ✓ model_metrics.csv")
    print("  ✓ cv_model_comparison.csv")
    print("  ✓ test_predictions.csv")
    print("  ✓ permutation_importance.csv")
    print("  ✓ best_model.joblib")
    print("  ✓ model_metadata.json")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
