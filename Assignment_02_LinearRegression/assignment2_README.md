# Assignment 2 – Simple & Multiple Linear Regression

## Objective
Implement Simple and Multiple Linear Regression to predict continuous variables (house prices).

## Dataset
**California Housing Dataset** — built into `sklearn`, no download needed.
- 20,640 samples | 8 features | Target: median house price (in $100,000s)
- Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

## What the Code Does
1. Loads and preprocesses the dataset (StandardScaler for feature scaling)
2. Checks for missing values
3. Fits a **Simple Linear Regression** model using only `MedInc` (median income)
4. Fits a **Multiple Linear Regression** model using all 8 features
5. Evaluates both models using **MSE**, **RMSE**, and **R² Score**
6. Visualizes: regression line, actual vs predicted plot, and model comparison bar chart

## How to Run
```bash
pip install scikit-learn matplotlib numpy
python assignment2_linear_regression.py
```

## Libraries Used
- `scikit-learn` — dataset, model, metrics
- `numpy` — numerical operations
- `matplotlib` — visualizations

## Output
- Printed metrics for both models in the terminal
- A saved plot: `assignment2_output.png`
