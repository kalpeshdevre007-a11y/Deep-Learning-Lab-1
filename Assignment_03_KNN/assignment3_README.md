# Assignment 3 – KNN Classification on Iris Dataset

## Objective
Implement the K-Nearest Neighbour algorithm to classify the Iris dataset. Print both correct and wrong predictions.

## Dataset
**Iris Dataset** — built into `sklearn`, no download needed.
- 150 samples | 4 features | 3 classes: Setosa, Versicolor, Virginica
- Features: sepal length, sepal width, petal length, petal width

## What the Code Does
1. Loads the Iris dataset
2. Splits into train/test sets and applies StandardScaler
3. Tests K values from 1 to 20 to find the best K
4. Trains final KNN model with best K
5. Prints **every prediction** — labelled CORRECT or WRONG
6. Prints accuracy, classification report
7. Visualizes: K vs Accuracy, Confusion Matrix, Scatter plot with wrong predictions circled

## How to Run
```bash
pip install scikit-learn matplotlib numpy
python assignment3_knn.py
```

## Libraries Used
- `scikit-learn` — dataset, KNN model, metrics
- `numpy` — numerical operations
- `matplotlib` — visualizations

## Output
- Full prediction table printed in terminal
- Plot saved as `assignment3_output.png`
