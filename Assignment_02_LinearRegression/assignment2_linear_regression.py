# Assignment 2: Simple and Multiple Linear Regression
# Dataset: California Housing (built-in from sklearn)
# Run: python assignment2_linear_regression.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X_all = housing.data        # 8 features
y = housing.target          # house price (in $100,000s)
feature_names = housing.feature_names

print(f"Dataset shape: {X_all.shape}")
print(f"Features: {feature_names}")

# ─────────────────────────────────────────────
# DATA PREPROCESSING
# ─────────────────────────────────────────────
# Handle missing values (none in this dataset, but good practice)
print(f"\nMissing values: {np.isnan(X_all).sum()}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# ─────────────────────────────────────────────
# SIMPLE LINEAR REGRESSION (1 feature: MedInc)
# ─────────────────────────────────────────────
print("\n--- SIMPLE LINEAR REGRESSION ---")
X_simple = X_scaled[:, 0].reshape(-1, 1)   # MedInc (median income)

X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train)
y_pred_simple = simple_model.predict(X_test_s)

mse_s  = mean_squared_error(y_test, y_pred_simple)
rmse_s = np.sqrt(mse_s)
r2_s   = r2_score(y_test, y_pred_simple)

print(f"Coefficient : {simple_model.coef_[0]:.4f}")
print(f"Intercept   : {simple_model.intercept_:.4f}")
print(f"MSE         : {mse_s:.4f}")
print(f"RMSE        : {rmse_s:.4f}")
print(f"R² Score    : {r2_s:.4f}")

# ─────────────────────────────────────────────
# MULTIPLE LINEAR REGRESSION (all 8 features)
# ─────────────────────────────────────────────
print("\n--- MULTIPLE LINEAR REGRESSION ---")
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)
y_pred_multi = multi_model.predict(X_test_m)

mse_m  = mean_squared_error(y_test_m, y_pred_multi)
rmse_m = np.sqrt(mse_m)
r2_m   = r2_score(y_test_m, y_pred_multi)

print("Coefficients per feature:")
for name, coef in zip(feature_names, multi_model.coef_):
    print(f"  {name:12s}: {coef:.4f}")
print(f"MSE      : {mse_m:.4f}")
print(f"RMSE     : {rmse_m:.4f}")
print(f"R² Score : {r2_m:.4f}")

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Assignment 2 – Linear Regression", fontsize=14, fontweight='bold')

# Plot 1: Simple Regression Line
sort_idx = np.argsort(X_test_s[:, 0])
axes[0].scatter(X_test_s, y_test, alpha=0.3, color='steelblue', label='Actual', s=10)
axes[0].plot(X_test_s[sort_idx], y_pred_simple[sort_idx], color='red', linewidth=2, label='Regression Line')
axes[0].set_xlabel("Median Income (scaled)")
axes[0].set_ylabel("House Price ($100k)")
axes[0].set_title("Simple Linear Regression")
axes[0].legend()

# Plot 2: Multiple Regression – Actual vs Predicted
axes[1].scatter(y_test_m, y_pred_multi, alpha=0.3, color='darkorange', s=10)
axes[1].plot([y_test_m.min(), y_test_m.max()],
             [y_test_m.min(), y_test_m.max()], 'r--', linewidth=2, label='Perfect Fit')
axes[1].set_xlabel("Actual Price")
axes[1].set_ylabel("Predicted Price")
axes[1].set_title("Multiple Regression: Actual vs Predicted")
axes[1].legend()

# Plot 3: Model Comparison Bar Chart
metrics = ['MSE', 'RMSE', 'R²']
simple_vals = [mse_s, rmse_s, r2_s]
multi_vals  = [mse_m, rmse_m, r2_m]
x = np.arange(len(metrics))
width = 0.35
axes[2].bar(x - width/2, simple_vals, width, label='Simple LR', color='steelblue')
axes[2].bar(x + width/2, multi_vals,  width, label='Multiple LR', color='darkorange')
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics)
axes[2].set_title("Model Comparison")
axes[2].legend()
for i, (sv, mv) in enumerate(zip(simple_vals, multi_vals)):
    axes[2].text(i - width/2, sv + 0.01, f"{sv:.2f}", ha='center', fontsize=8)
    axes[2].text(i + width/2, mv + 0.01, f"{mv:.2f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("assignment2_output.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as assignment2_output.png")
