# Assignment 1 — PCA: Dimensionality Reduction

## Objective
Reduce the dimensions of a high-dimensional dataset using Principal Component Analysis (PCA).

## Dataset
**Breast Cancer Wisconsin Dataset** (built-in via `sklearn.datasets`)
- 569 samples, 30 features
- Binary classification: Malignant vs Benign

## Steps Performed
1. Loaded the dataset and explored its shape
2. Standardized the data using `StandardScaler` (zero mean, unit variance)
3. Applied PCA to all 30 components
4. Constructed a **Scree Plot** to visualize explained variance per component
5. Reduced dimensions to **2 components** and visualized the data in 2D

## How to Run
```bash
python assignment1_pca.py
```

## Libraries Used
- `scikit-learn` — PCA, StandardScaler, dataset
- `numpy` — numerical operations
- `matplotlib` — plotting

## Output
- `scree_plot.png` — Bar chart of variance per component + cumulative variance curve
- `pca_2d_visualization.png` — 2D scatter plot showing class separation after PCA
