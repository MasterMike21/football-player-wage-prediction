# Football Player Wage Intelligence & Valuation

A DS + DA project that studies football player wages, predicts wage from player characteristics, explains model behavior, and provides analytics outputs for Power BI.

## Project objective

Answer two related questions:

1. Which player attributes are associated with higher wages in the supplied dataset?
2. How does each player's observed wage compare with the model's estimated wage given the available features?

## Dataset

The project uses the supplied `SalaryPrediction.csv`.

Original fields:
- Wage
- Age
- Club
- League
- Nation
- Position
- Apps
- Caps

The source dataset does not contain player names. The project therefore creates a synthetic `Player_ID` after cleaning for record-level reporting. This identifier is not used as a model feature.

Cleaning:
- Wage strings are converted to numeric.
- The supplied source spelling `Midfilder` is normalized to `Midfielder`.
- Exact duplicate rows are removed.
- Three interpretable features are created:
  - Caps_per_App
  - Apps_per_Year
  - International_Exposure

## Machine learning

Models:
- Median baseline
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Categorical variables are encoded with OneHotEncoder inside a Scikit-learn pipeline.

The model is selected using test-set RMSE, with MAE as a tiebreaker.

Metrics:
- MAE
- RMSE
- R²

Explainability:
- Permutation importance on the original human-readable features.

## Data analytics

SQL:
- Average wage by position/league
- Top-paid players
- Window-function ranking within leagues
- Age-band analysis
- International exposure analysis
- Appearance-band analysis

Power BI:
- Wage overview
- Player valuation
- Model analytics

See `powerbi/POWER_BI_GUIDE.md`.

## Run locally

### 1. Create and activate an environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the project

```bash
python train.py
```

This creates:
- `artifacts/model_metrics.csv`
- `artifacts/best_model.joblib`
- `artifacts/test_predictions.csv`
- `artifacts/permutation_importance.csv`
- `artifacts/model_metadata.json`

### 4. Build the SQLite database

```bash
python sql/build_sqlite.py
```

Database:
`artifacts/football_wages.db`

### 5. Launch the dashboard

```bash
streamlit run app.py
```

## Resume-safe project description

**Football Player Wage Intelligence & Valuation**  
`Python | SQL | Power BI | Scikit-learn | Pandas | Machine Learning`

- Built an end-to-end analytics and regression pipeline to analyze football player wages using SQL, Python and engineered player-experience features.
- Compared Linear Regression, Random Forest and Gradient Boosting using MAE, RMSE and R², selecting the strongest model through held-out evaluation.
- Developed model-relative wage benchmarking and an interactive analytics dashboard covering league, club, position and player-level wage patterns.

Only use the numbers from `artifacts/model_metrics.csv` after running the project yourself.

## Important interview note

This project demonstrates association and prediction, not causal inference. A player having a high feature-importance score or a wage above the model estimate does not prove that the feature caused the wage or that the player is objectively over/underpaid.
