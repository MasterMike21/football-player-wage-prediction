# Interview Preparation — Football Player Wage Intelligence

## 60-second explanation

"I built a football wage intelligence project to study which player characteristics are associated with wages and to estimate a player's wage from the features available in the dataset. I cleaned the wage field, normalized categorical data, created interpretable exposure features, performed EDA and compared multiple regression models using MAE, RMSE and R². I also added SQL-based analysis and a Power BI layer. Finally, I used permutation importance and model-relative wage differences to make the predictions more interpretable."

## Questions you should be ready for

### Data
1. Where did the data come from?
2. How many rows and columns?
3. What is the target variable?
4. Were there missing values?
5. Did you find duplicates?
6. Why did you normalize `Midfilder`?

### EDA
7. Is wage normally distributed?
8. Why is a monetary target often right-skewed?
9. What relationships did you inspect?
10. Does correlation imply causation?

### Preprocessing
11. Why use OneHotEncoder rather than LabelEncoder?
12. Why put preprocessing inside a Pipeline?
13. How do you handle unseen categories at inference time?

### Modeling
14. Why is this regression?
15. Why use a baseline?
16. Why compare Linear Regression, Random Forest and Gradient Boosting?
17. What is overfitting?
18. Why use a held-out test set?
19. What is cross-validation and how would you add it?

### Metrics
20. Difference between MAE, RMSE and R²?
21. Why might RMSE be larger than MAE?
22. When can R² be misleading?

### Feature engineering
23. Why create Caps_per_App?
24. What does International_Exposure represent?
25. Could Apps_per_Year introduce an assumption?
26. What would you do if Apps = 0?

### Explainability
27. What is permutation importance?
28. How is it different from tree impurity importance?
29. Does feature importance prove causality? (No.)

### Business
30. How could a club use this analysis?
31. How would you make the dashboard useful to a recruiter/manager?
32. What would you do with players whose observed wage differs substantially from the model estimate?

## Honest limitations to mention

- The dataset contains only the listed player-level attributes; richer performance statistics could improve the model.
- Club/league/nation variables may capture contextual effects that are difficult to separate cleanly.
- A model-relative wage difference is an analytical comparison, not a market valuation truth.
- Observational data alone cannot establish causal relationships.
