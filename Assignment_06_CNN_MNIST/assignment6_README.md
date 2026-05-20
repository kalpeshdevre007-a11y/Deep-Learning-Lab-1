# Assignment 6: CNN Multiclass Classifier (MNIST)

## Objective
Build a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset.

## Dataset
**MNIST** — Built-in from `keras.datasets`
- 60,000 training + 10,000 test images
- 28×28 grayscale images, 10 classes (digits 0–9)
- No download required — loads automatically via Keras

## What This Assignment Covers
| Part | Task |
|------|------|
| A | Data Preprocessing (normalize, reshape, one-hot encode) |
| B | Define CNN Model and train it |
| C | Evaluate using accuracy, loss, and confusion matrix |

## CNN Architecture
```
Input (28x28x1)
  → Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout(0.25)
  → Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout(0.25)
  → Flatten
  → Dense(256) + BatchNorm + Dropout(0.5)
  → Dense(10, softmax)
```

## How to Run
```bash
python assignment6_cnn_mnist.py
```

### Requirements
```
tensorflow
numpy
matplotlib
scikit-learn
```

## Output
- Console: model summary, training logs, accuracy, classification report
- Plot: `assignment6_output.png` with:
  1. Training/Validation Accuracy curve
  2. Training/Validation Loss curve
  3. Confusion Matrix (10×10)
  4. Sample correct predictions
  5. Sample wrong predictions
  6. Per-class accuracy bar chart

## Expected Results
- Test Accuracy: ~99%
- Training Time: ~2–3 minutes on CPU

## GitHub Folder Structure
```
Assignment_06_CNN_MNIST/
├── assignment6_cnn_mnist.py
└── README.md
```
