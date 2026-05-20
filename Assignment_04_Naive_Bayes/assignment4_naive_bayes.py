# Assignment 4: Naive Bayes Classifier
# Dataset: Wine Dataset (built-in from sklearn)
# Run: python assignment4_naive_bayes.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
print("Loading Wine dataset...")
wine = load_wine()
X = wine.data
y = wine.target
class_names = wine.target_names
feature_names = wine.feature_names

print(f"Dataset shape : {X.shape}")
print(f"Classes       : {class_names}")
print(f"Features      : {len(feature_names)} chemical properties")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
print("\nPreprocessing data...")
print(f"Missing values: {np.isnan(X).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# TRAIN NAIVE BAYES MODEL
# ─────────────────────────────────────────────
print("\nTraining Gaussian Naive Bayes classifier...")
model = GaussianNB()
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# PREDICTIONS ON MULTIPLE TEST SETS
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ─────────────────────────────────────────────
# PRINT PREDICTIONS — correct & wrong
# ─────────────────────────────────────────────
print("\n--- PREDICTION RESULTS ON TEST DATA ---")
print(f"{'#':<5}{'Actual':<15}{'Predicted':<15}{'Confidence':<12}{'Result'}")
print("-" * 60)
for i, (actual, predicted, prob) in enumerate(zip(y_test, y_pred, y_prob)):
    confidence = prob[predicted] * 100
    result = "✓ CORRECT" if actual == predicted else "✗ WRONG"
    print(f"{i:<5}{class_names[actual]:<15}{class_names[predicted]:<15}{confidence:<12.1f}%  {result}")

correct = np.sum(y_test == y_pred)
wrong   = np.sum(y_test != y_pred)
print(f"\nTotal Correct : {correct}")
print(f"Total Wrong   : {wrong}")
print(f"Accuracy      : {accuracy_score(y_test, y_pred)*100:.2f}%")

# ─────────────────────────────────────────────
# FEW SPECIFIC TEST SAMPLES (as required)
# ─────────────────────────────────────────────
print("\n--- SAMPLE-WISE EVALUATION (5 test samples) ---")
sample_indices = [0, 1, 2, 3, 4]
for idx in sample_indices:
    sample = X_test[idx].reshape(1, -1)
    pred   = model.predict(sample)[0]
    actual = y_test[idx]
    prob   = model.predict_proba(sample)[0]
    print(f"Sample {idx}: Actual={class_names[actual]}, "
          f"Predicted={class_names[pred]}, "
          f"Confidence={prob[pred]*100:.1f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Assignment 4 – Naive Bayes on Wine Dataset", fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")
axes[0].set_xticks([0, 1, 2])
axes[0].set_yticks([0, 1, 2])
axes[0].set_xticklabels(class_names, rotation=15)
axes[0].set_yticklabels(class_names)
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)

# Plot 2: Prediction Confidence Bar Chart
confidences = [y_prob[i][y_pred[i]] * 100 for i in range(len(y_pred))]
colors_bar  = ['green' if a == p else 'red' for a, p in zip(y_test, y_pred)]
axes[1].bar(range(len(confidences)), confidences, color=colors_bar, alpha=0.7)
axes[1].axhline(y=50, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel("Test Sample Index")
axes[1].set_ylabel("Confidence (%)")
axes[1].set_title("Prediction Confidence\n(Green=Correct, Red=Wrong)")
axes[1].set_ylim(0, 105)

# Plot 3: Class distribution in test set
unique, counts = np.unique(y_test, return_counts=True)
axes[2].bar(class_names[unique], counts, color=['steelblue', 'darkorange', 'green'], alpha=0.8)
axes[2].set_title("Test Set Class Distribution")
axes[2].set_xlabel("Wine Class")
axes[2].set_ylabel("Count")
for i, (u, c) in enumerate(zip(unique, counts)):
    correct_in_class = np.sum((y_test == u) & (y_pred == u))
    axes[2].text(i, c + 0.2, f"{correct_in_class}/{c} correct", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("assignment4_output.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as assignment4_output.png")
