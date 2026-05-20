# Assignment 1: PCA - Dimensionality Reduction
# Dataset: Breast Cancer Wisconsin (built-in from sklearn)
# Run: python assignment1_pca.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── 1. Load Dataset ──────────────────────────────────────────────
data = load_breast_cancer()
X = data.data        # 30 features
y = data.target      # 0 = malignant, 1 = benign
print(f"Original dataset shape: {X.shape}")  # (569, 30)

# ── 2. Standardization (Pre-processing) ─────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data standardized: mean ~0, std ~1")

# ── 3. PCA - reduce to all components to plot scree ─────────────
pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# ── 4. Scree Plot ────────────────────────────────────────────────
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='steelblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.xticks(range(1, len(explained_variance) + 1, 2))

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='darkorange')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance')
plt.legend()

plt.tight_layout()
plt.savefig('scree_plot.png')
plt.show()
print("Scree plot saved as scree_plot.png")

# ── 5. PCA - reduce to 2 components ─────────────────────────────
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)
print(f"\nReduced dataset shape: {X_pca.shape}")  # (569, 2)
print(f"Variance explained by 2 components: {pca_2d.explained_variance_ratio_.sum():.2%}")

# ── 6. 2D Visualization ──────────────────────────────────────────
plt.figure(figsize=(8, 6))
colors = ['red', 'blue']
labels = ['Malignant', 'Benign']

for i, (color, label) in enumerate(zip(colors, labels)):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=color, label=label, alpha=0.6, edgecolors='k', linewidths=0.3)

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA - 2D Visualization of Breast Cancer Dataset')
plt.legend()
plt.tight_layout()
plt.savefig('pca_2d_visualization.png')
plt.show()
print("2D PCA plot saved as pca_2d_visualization.png")

print("\n--- PCA Summary ---")
print(f"Original dimensions : 30")
print(f"Reduced dimensions  : 2")
print(f"Total variance kept : {pca_2d.explained_variance_ratio_.sum():.2%}")
