# Assignment 5: Decision Trees, Random Forest, and AdaBoost
# Dataset: Iris Dataset (sklearn built-in)
# Run: python assignment5_decision_trees.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Load and Split Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  Assignment 5: Decision Trees, Random Forest, AdaBoost")
print("=" * 60)

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDataset: Iris ({len(X)} samples, {X.shape[1]} features, {len(class_names)} classes)")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ─────────────────────────────────────────────
# 2. Basic Decision Tree (no pruning)
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("PART A & B: Decision Tree (No Pruning)")
print("─" * 60)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc  = accuracy_score(y_test,  dt.predict(X_test))

print(f"Training Accuracy : {train_acc * 100:.2f}%")
print(f"Testing  Accuracy : {test_acc  * 100:.2f}%")
print(f"Tree Depth        : {dt.get_depth()}")
print(f"Number of Leaves  : {dt.get_n_leaves()}")

if train_acc > test_acc + 0.05:
    print(">> Overfitting detected (train acc >> test acc)")

# ─────────────────────────────────────────────
# 3. Cost Complexity Pruning
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("PART D: Cost Complexity Pruning (to fix overfitting)")
print("─" * 60)

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]

train_scores, test_scores = [], []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test,  clf.predict(X_test)))

best_idx   = np.argmax(test_scores)
best_alpha = ccp_alphas[best_idx]

pruned_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_dt.fit(X_train, y_train)

pruned_train = accuracy_score(y_train, pruned_dt.predict(X_train))
pruned_test  = accuracy_score(y_test,  pruned_dt.predict(X_test))

print(f"Best ccp_alpha    : {best_alpha:.4f}")
print(f"Training Accuracy : {pruned_train * 100:.2f}%")
print(f"Testing  Accuracy : {pruned_test  * 100:.2f}%")
print(f"Tree Depth        : {pruned_dt.get_depth()}")
print(f"Number of Leaves  : {pruned_dt.get_n_leaves()}")
print(classification_report(y_test, pruned_dt.predict(X_test), target_names=class_names))

# ─────────────────────────────────────────────
# 4. Random Forest
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("PART E: Random Forest (overcome overfitting)")
print("─" * 60)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_train = accuracy_score(y_train, rf.predict(X_train))
rf_test  = accuracy_score(y_test,  rf.predict(X_test))

print(f"Training Accuracy : {rf_train * 100:.2f}%")
print(f"Testing  Accuracy : {rf_test  * 100:.2f}%")
print(f"Number of Trees   : {rf.n_estimators}")
print(classification_report(y_test, rf.predict(X_test), target_names=class_names))

print("Feature Importances (Random Forest):")
for name, imp in zip(feature_names, rf.feature_importances_):
    bar = "█" * int(imp * 40)
    print(f"  {name:<30} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────
# 5. AdaBoost
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("PART F: AdaBoost on Decision Stumps")
print("─" * 60)

stump = DecisionTreeClassifier(max_depth=1)
ada = AdaBoostClassifier(estimator=stump, n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

ada_train = accuracy_score(y_train, ada.predict(X_train))
ada_test  = accuracy_score(y_test,  ada.predict(X_test))

print(f"Training Accuracy : {ada_train * 100:.2f}%")
print(f"Testing  Accuracy : {ada_test  * 100:.2f}%")
print(f"Number of Stumps  : {ada.n_estimators}")
print(classification_report(y_test, ada.predict(X_test), target_names=class_names))

# ─────────────────────────────────────────────
# 6. Comparison Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<30} {'Train Acc':>10} {'Test Acc':>10}")
print("─" * 55)
print(f"{'Decision Tree (unpruned)':<30} {train_acc*100:>9.2f}% {test_acc*100:>9.2f}%")
print(f"{'Decision Tree (pruned)':<30} {pruned_train*100:>9.2f}% {pruned_test*100:>9.2f}%")
print(f"{'Random Forest':<30} {rf_train*100:>9.2f}% {rf_test*100:>9.2f}%")
print(f"{'AdaBoost':<30} {ada_train*100:>9.2f}% {ada_test*100:>9.2f}%")

# ─────────────────────────────────────────────
# 7. Visualizations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Assignment 5: Decision Trees, Random Forest & AdaBoost", fontsize=14, fontweight='bold')

# Plot 1: Decision Tree (pruned)
plot_tree(pruned_dt, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, ax=axes[0, 0], fontsize=7)
axes[0, 0].set_title("Pruned Decision Tree")

# Plot 2: Pruning curve
axes[0, 1].plot(ccp_alphas, train_scores, marker='o', label='Train', color='steelblue')
axes[0, 1].plot(ccp_alphas, test_scores,  marker='s', label='Test',  color='tomato')
axes[0, 1].axvline(best_alpha, color='green', linestyle='--', label=f'Best α={best_alpha:.4f}')
axes[0, 1].set_xlabel("ccp_alpha")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("Cost Complexity Pruning Curve")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature Importance (Random Forest)
importances = rf.feature_importances_
axes[0, 2].barh(feature_names, importances, color='steelblue')
axes[0, 2].set_xlabel("Importance")
axes[0, 2].set_title("Random Forest Feature Importances")
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, rf.predict(X_test))
im = axes[1, 0].imshow(cm_rf, cmap='Blues')
axes[1, 0].set_xticks(range(3)); axes[1, 0].set_yticks(range(3))
axes[1, 0].set_xticklabels(class_names, rotation=45)
axes[1, 0].set_yticklabels(class_names)
for i in range(3):
    for j in range(3):
        axes[1, 0].text(j, i, cm_rf[i, j], ha='center', va='center', fontsize=12,
                        color='white' if cm_rf[i, j] > cm_rf.max()/2 else 'black')
axes[1, 0].set_title("Confusion Matrix (Random Forest)")
axes[1, 0].set_xlabel("Predicted"); axes[1, 0].set_ylabel("Actual")

# Plot 5: AdaBoost staged accuracy
ada_test_staged = [accuracy_score(y_test, pred) for pred in ada.staged_predict(X_test)]
ada_train_staged = [accuracy_score(y_train, pred) for pred in ada.staged_predict(X_train)]
axes[1, 1].plot(ada_train_staged, label='Train', color='steelblue')
axes[1, 1].plot(ada_test_staged,  label='Test',  color='tomato')
axes[1, 1].set_xlabel("Number of Estimators")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_title("AdaBoost: Accuracy vs Number of Stumps")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Model comparison bar chart
models  = ['DT\n(unpruned)', 'DT\n(pruned)', 'Random\nForest', 'AdaBoost']
tr_accs = [train_acc, pruned_train, rf_train, ada_train]
te_accs = [test_acc,  pruned_test,  rf_test,  ada_test]
x = np.arange(len(models))
axes[1, 2].bar(x - 0.2, [a*100 for a in tr_accs], 0.4, label='Train', color='steelblue')
axes[1, 2].bar(x + 0.2, [a*100 for a in te_accs], 0.4, label='Test',  color='tomato')
axes[1, 2].set_xticks(x); axes[1, 2].set_xticklabels(models)
axes[1, 2].set_ylabel("Accuracy (%)")
axes[1, 2].set_title("Model Comparison")
axes[1, 2].legend()
axes[1, 2].set_ylim(80, 105)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("assignment5_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("\nPlot saved as assignment5_output.png")
print("\nAssignment 5 Complete!")
