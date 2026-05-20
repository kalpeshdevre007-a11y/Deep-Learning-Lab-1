# Assignment 3: K-Nearest Neighbour (KNN) Classification
# Dataset: Iris (built-in from sklearn)
# Run: python assignment3_knn.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
feature_names = iris.feature_names

print(f"Dataset shape : {X.shape}")
print(f"Classes       : {class_names}")
print(f"Features      : {feature_names}")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# FIND BEST K
# ─────────────────────────────────────────────
print("\nFinding best K value...")
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, knn.predict(X_test)))

best_k = k_range[np.argmax(k_scores)]
print(f"Best K = {best_k} with accuracy = {max(k_scores)*100:.2f}%")

# ─────────────────────────────────────────────
# TRAIN FINAL MODEL
# ─────────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# ─────────────────────────────────────────────
# CORRECT & WRONG PREDICTIONS (as required)
# ─────────────────────────────────────────────
print("\n--- PREDICTION RESULTS ---")
print(f"{'Index':<8}{'Actual':<20}{'Predicted':<20}{'Result'}")
print("-" * 60)
correct = 0
wrong   = 0
for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
    result = "✓ CORRECT" if actual == predicted else "✗ WRONG"
    if actual == predicted:
        correct += 1
    else:
        wrong += 1
    print(f"{i:<8}{class_names[actual]:<20}{class_names[predicted]:<20}{result}")

print(f"\nTotal Correct    : {correct}")
print(f"Total Wrong      : {wrong}")
print(f"Accuracy         : {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Assignment 3 – KNN on Iris Dataset", fontsize=14, fontweight='bold')

# Plot 1: K vs Accuracy
axes[0].plot(k_range, k_scores, marker='o', color='steelblue', linewidth=2)
axes[0].axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0].set_xlabel("K Value")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("K vs Accuracy")
axes[0].legend()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")
axes[1].set_xticks([0, 1, 2])
axes[1].set_yticks([0, 1, 2])
axes[1].set_xticklabels(class_names, rotation=15)
axes[1].set_yticklabels(class_names)
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)

# Plot 3: Scatter – Sepal features colored by class
colors = ['red', 'green', 'blue']
for cls in range(3):
    idx = y_test == cls
    axes[2].scatter(X_test[idx, 0], X_test[idx, 1],
                    color=colors[cls], label=class_names[cls], alpha=0.7, s=60)
# Mark wrong predictions
wrong_idx = y_test != y_pred
axes[2].scatter(X_test[wrong_idx, 0], X_test[wrong_idx, 1],
                facecolors='none', edgecolors='black', s=150, linewidths=2, label='Wrong prediction')
axes[2].set_xlabel("Sepal Length (scaled)")
axes[2].set_ylabel("Sepal Width (scaled)")
axes[2].set_title("Test Data – Wrong Predictions Circled")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("assignment3_output.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as assignment3_output.png")
