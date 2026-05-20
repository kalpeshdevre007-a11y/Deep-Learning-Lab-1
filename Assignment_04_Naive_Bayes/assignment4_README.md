# Assignment 4 – Naïve Bayesian Classifier

## Objective
Implement the Naïve Bayesian Classifier on a training dataset. Compute accuracy and evaluate on multiple test data sets.

## Dataset
**Wine Dataset** — built into `sklearn`, no download needed.
- 178 samples | 13 chemical features | 3 wine classes
- Features: alcohol, malic acid, ash, magnesium, flavanoids, etc.

## What the Code Does
1. Loads the Wine dataset and checks for missing values
2. Splits into train/test sets and applies StandardScaler
3. Trains a **Gaussian Naive Bayes** classifier
4. Prints every prediction with confidence % — labelled CORRECT or WRONG
5. Evaluates on 5 specific test samples individually
6. Prints full accuracy and classification report
7. Visualizes: Confusion Matrix, Confidence bar chart, Class distribution

## How to Run
```bash
pip install scikit-learn matplotlib numpy
python assignment4_naive_bayes.py
```

## Libraries Used
- `scikit-learn` — dataset, GaussianNB model, metrics
- `numpy` — numerical operations
- `matplotlib` — visualizations

## Output
- Prediction table with confidence scores printed in terminal
- Plot saved as `assignment4_output.png`
