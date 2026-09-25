# ⚽ Football Player Wage Prediction

An end-to-end **Data Science and Machine Learning project** that analyzes football player wages and builds regression models to estimate player wages from available player characteristics.

The project covers the complete machine learning workflow:

**Data Cleaning → Exploratory Data Analysis → Feature Engineering → Model Training → Cross-Validation → Model Selection → Final Test Evaluation → Model Explainability → Prediction Analysis**

---

## 🎯 Project Objective

The main objective is to investigate the factors associated with football player wages and build a machine learning model capable of estimating wage from player-level attributes.

The project focuses on two questions:

1. **Which player characteristics are associated with differences in wages?**
2. **How accurately can machine learning models estimate player wages from the available data?**

> This project focuses on prediction and association, not causal inference.

---

## 📊 Dataset

The project uses the supplied `SalaryPrediction.csv` dataset.

### Original Features

- `Wage`
- `Age`
- `Club`
- `League`
- `Nation`
- `Position`
- `Apps`
- `Caps`

The source dataset does **not** contain player names.

For record-level prediction analysis, the project creates a synthetic `Player_ID`. This identifier is used only for tracking/reporting and is **never used as a machine-learning feature**.

---

## 🧹 Data Preparation

The preprocessing pipeline performs the following operations:

### Wage Cleaning

The original wage values are stored as strings containing commas.

Example:

```text
46,427,000
