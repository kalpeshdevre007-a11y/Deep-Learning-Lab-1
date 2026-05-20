# Assignment 5: Decision Trees, Random Forest & AdaBoost

## Objective
Learn Decision Trees for classification, apply pruning, and use ensemble methods (Random Forest and AdaBoost) to overcome overfitting.

## Dataset
**Iris Dataset** — Built-in from `sklearn.datasets`
- 150 samples, 4 features, 3 classes (Setosa, Versicolor, Virginica)
- No download required — loads automatically

## What This Assignment Covers
| Part | Task |
|------|------|
| A | Split dataset into training and test sets |
| B | Build a basic Decision Tree |
| C | Check model performance on train and test data |
| D | Apply Cost Complexity Pruning to fix overfitting |
| E | Apply Random Forest to overcome overfitting |
| F | Apply AdaBoost ensemble method on Decision Stumps |

## How to Run
```bash
python assignment5_decision_trees.py
```

### Requirements
```
numpy
matplotlib
scikit-learn
```

## Output
- Console: accuracy tables, classification reports, feature importances
- Plot: `assignment5_output.png` with 6 visualizations:
  1. Pruned Decision Tree diagram
  2. Cost Complexity Pruning curve
  3. Random Forest feature importances
  4. Random Forest confusion matrix
  5. AdaBoost staged accuracy curve
  6. Model comparison bar chart

## Key Concepts
- **Overfitting**: Unpruned tree has high train accuracy but lower test accuracy
- **Pruning (ccp_alpha)**: Reduces tree complexity → better generalization
- **Random Forest**: Ensemble of 100 trees → more robust predictions
- **AdaBoost**: Boosts weak learners (depth-1 stumps) sequentially

## GitHub Folder Structure
```
Assignment_05_Decision_Trees/
├── assignment5_decision_trees.py
└── README.md
```
