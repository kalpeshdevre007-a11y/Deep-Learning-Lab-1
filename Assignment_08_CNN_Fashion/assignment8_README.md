# Assignment 8: Fashion Clothing Classifier using CNN

## Objective
Build a CNN classifier to classify fashion clothing images into 10 categories using the Fashion-MNIST dataset.

## Dataset
**Fashion-MNIST** — Built-in from `keras.datasets`
- 60,000 training + 10,000 test images
- 28×28 grayscale images, 10 clothing categories
- No download required — loads automatically via Keras

| Label | Category     |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

## What This Assignment Covers
| Step | Task |
|------|------|
| 1 | Load and preprocess Fashion-MNIST (normalize, reshape, one-hot encode) |
| 2 | Check class distribution |
| 3 | Build 3-block CNN architecture |
| 4 | Train with EarlyStopping + ReduceLROnPlateau callbacks |
| 5 | Evaluate with accuracy, loss, classification report |
| 6 | Visualize results with 6 plots |

## CNN Architecture
```
Input (28x28x1 Grayscale)
  → Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout(0.25)
  → Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout(0.25)
  → Conv2D(128) + BatchNorm + Dropout(0.25)
  → Flatten
  → Dense(256) + BatchNorm + Dropout(0.5)
  → Dense(10, softmax)
```

## How to Run
```bash
python assignment8_cnn_fashion.py
```

### Requirements
```
tensorflow
numpy
matplotlib
scikit-learn
```

## Output
- Console: model summary, training logs, classification report
- Plot: `assignment8_output.png` with:
  1. Sample images from all 10 clothing categories
  2. Training/Validation Accuracy curve
  3. Training/Validation Loss curve
  4. Confusion Matrix (10×10)
  5. Per-class accuracy (color-coded: green/orange/red)
  6. Sample correct vs wrong predictions

## Expected Results
- Test Accuracy: ~91–93%
- Training Time: ~5–8 minutes on CPU (early stopping may reduce this)

## GitHub Folder Structure
```
Assignment_08_CNN_Fashion/
├── assignment8_cnn_fashion.py
└── README.md
```
