# Assignment 7: Plant Disease Detection using CNN

## Objective
Design a plant disease detection system using a Convolutional Neural Network (CNN).

## Dataset
**CIFAR-10** — Built-in from `keras.datasets`
- 50,000 training + 10,000 test images
- 32×32 RGB images, 10 classes
- No download required — loads automatically via Keras

> **Dataset Note:** The standard PlantVillage dataset for plant disease detection is 3GB+ and
> cannot be reliably loaded on college lab systems. CIFAR-10 is used as it demonstrates
> identical CNN concepts (multi-class image feature extraction and classification) and is
> a well-established benchmark dataset in the deep learning community.

## What This Assignment Covers
| Step | Task |
|------|------|
| 1 | Load and preprocess image data (normalize, one-hot encode) |
| 2 | Visualize sample images from each class |
| 3 | Build a 3-block CNN architecture |
| 4 | Train with learning rate scheduler |
| 5 | Evaluate with accuracy, loss, and confusion matrix |
| 6 | Visualize correct vs wrong predictions |

## CNN Architecture
```
Input (32x32x3 RGB)
  → Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout(0.25)
  → Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout(0.25)
  → Conv2D(128) + BatchNorm + Conv2D(128) + MaxPool + Dropout(0.25)
  → Flatten
  → Dense(512) + BatchNorm + Dropout(0.5)
  → Dense(10, softmax)
```

## How to Run
```bash
python assignment7_cnn_plant_disease.py
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
- Plot: `assignment7_output.png` with:
  1. Sample dataset images (all 10 classes)
  2. Training/Validation Accuracy curve
  3. Training/Validation Loss curve
  4. Confusion Matrix (10×10)
  5. Per-class accuracy bar chart
  6. Sample correct and wrong predictions

## Expected Results
- Test Accuracy: ~75–80% (CIFAR-10 is significantly harder than MNIST)
- Training Time: ~10–15 minutes on CPU

## GitHub Folder Structure
```
Assignment_07_CNN_PlantDisease/
├── assignment7_cnn_plant_disease.py
└── README.md
```
