# Power BI Dashboard Guide

The project exports two especially useful files:

- `data/cleaned_players.csv`
- `artifacts/test_predictions.csv`

For the dashboard, use `cleaned_players.csv` for core analysis and `test_predictions.csv` for player-level model outputs. The supplied dataset contains no player-name field, so the project creates a synthetic `Player_ID` only for record identification.

## Suggested report pages

### 1. Wage Overview
Cards:
- Player Count
- Average Wage
- Median Wage
- Maximum Wage

Charts:
- Average Wage by Position
- Average Wage by League
- Average Wage by Club
- Wage Distribution
- Wage by Age

### 2. Player Valuation
Use:
- Player_ID / Club / League slicers
- Actual Wage
- Predicted Wage
- Wage Difference
- Expected Wage Ratio

Interpret the difference as:
`Observed Wage - Model Estimated Wage`

Do not label a player as objectively "overpaid" or "underpaid"; present it as a model-relative comparison.

### 3. Model Analytics
Cards:
- MAE
- RMSE
- R²

Charts:
- Actual vs Predicted Wage
- Permutation Importance
- Absolute Error Distribution

## Helpful DAX measures

```DAX
Average Wage = AVERAGE(cleaned_players[Wage])

Median Wage = MEDIAN(cleaned_players[Wage])

Player Count = DISTINCTCOUNT(cleaned_players[player_id])

Average Age = AVERAGE(cleaned_players[Age])
```

If you load `test_predictions.csv` as a separate table, create:

```DAX
Average Prediction Error =
AVERAGE(test_predictions[Absolute_Error])

Mean Wage Difference =
AVERAGE(test_predictions[Wage_Difference])
```

## Recommended design

Keep the dashboard analytical rather than decorative. Use slicers for:
- League
- Club
- Position
- Nation
- Age band

A useful final page is "Model Analytics", which lets an interviewer see that the dashboard is backed by an evaluated ML model rather than only descriptive charts.
