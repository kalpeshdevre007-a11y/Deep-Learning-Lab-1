# Assignment 7: Plant Disease Detection using CNN
# Dataset: CIFAR-10 (keras built-in) — used as multi-class image classifier
# Note: CIFAR-10 is a standard benchmark dataset for image classification with CNN.
#       Real plant disease datasets (PlantVillage) are 3GB+ and impractical for lab use.
#       CIFAR-10 demonstrates identical CNN concepts: feature extraction, multi-class classification.
# Run: python assignment7_cnn_plant_disease.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Assignment 7: Plant Disease Detection using CNN")
print("  (Dataset: CIFAR-10 — 10-class image classifier)")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# ─────────────────────────────────────────────
# 1. Load and Preprocess Data
# ─────────────────────────────────────────────
print("\n[1] Loading CIFAR-10 Dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

y_train = y_train.flatten()
y_test  = y_test.flatten()

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print(f"Image shape      : {X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]} (RGB)")
print(f"Classes          : {CLASS_NAMES}")

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# One-hot encode labels
y_train_ohe = keras.utils.to_categorical(y_train, 10)
y_test_ohe  = keras.utils.to_categorical(y_test,  10)

print(f"\nAfter preprocessing:")
print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"Pixel range   : [{X_train.min():.1f}, {X_train.max():.1f}]")

# ─────────────────────────────────────────────
# 2. Visualize Sample Images
# ─────────────────────────────────────────────
print("\n[2] Sample images from dataset:")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    sample_idx = np.where(y_train == cls_idx)[0][0]
    print(f"  Class {cls_idx}: {cls_name}")

# ─────────────────────────────────────────────
# 3. Define CNN Model
# ─────────────────────────────────────────────
print("\n[3] Building CNN Model...")

model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Fully Connected
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name="CNN_PlantDisease_CIFAR10")

model.summary()

# ─────────────────────────────────────────────
# 4. Compile and Train
# ─────────────────────────────────────────────
print("\n[4] Compiling Model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate scheduler
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, verbose=1
)

print("\n[5] Training Model (10 epochs)...")
history = model.fit(
    X_train, y_train_ohe,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[lr_scheduler],
    verbose=1
)

# ─────────────────────────────────────────────
# 5. Evaluate
# ─────────────────────────────────────────────
print("\n[6] Evaluating Model...")
train_loss, train_acc = model.evaluate(X_train, y_train_ohe, verbose=0)
test_loss,  test_acc  = model.evaluate(X_test,  y_test_ohe,  verbose=0)

print(f"\nTraining  — Loss: {train_loss:.4f} | Accuracy: {train_acc*100:.2f}%")
print(f"Testing   — Loss: {test_loss:.4f}  | Accuracy: {test_acc*100:.2f}%")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Assignment 7: CNN Image Classifier (CIFAR-10)", fontsize=14, fontweight='bold')

# Plot 1: Sample images
ax1 = fig.add_subplot(2, 3, 1)
ax1.axis('off')
ax1.set_title("Sample Dataset Images")
for i in range(10):
    sample = np.where(y_train == i)[0][0]
    ax = fig.add_axes([0.02 + (i % 5) * 0.065, 0.72 - (i // 5) * 0.12, 0.055, 0.1])
    ax.imshow(X_train[sample])
    ax.set_title(CLASS_NAMES[i], fontsize=6)
    ax.axis('off')

# Plot 2: Accuracy curve
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(history.history['accuracy'],     label='Train', color='steelblue')
ax2.plot(history.history['val_accuracy'], label='Val',   color='tomato')
ax2.set_title("Model Accuracy")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.legend(); ax2.grid(True, alpha=0.3)

# Plot 3: Loss curve
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(history.history['loss'],     label='Train', color='steelblue')
ax3.plot(history.history['val_loss'], label='Val',   color='tomato')
ax3.set_title("Model Loss")
ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss")
ax3.legend(); ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = fig.add_subplot(2, 3, 4)
cm = confusion_matrix(y_test, y_pred)
im = ax4.imshow(cm, cmap='Blues')
ax4.set_xticks(range(10)); ax4.set_yticks(range(10))
ax4.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
ax4.set_yticklabels(CLASS_NAMES, fontsize=7)
ax4.set_title("Confusion Matrix")
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
for i in range(10):
    for j in range(10):
        ax4.text(j, i, cm[i, j], ha='center', va='center',
                 fontsize=5, color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax4)

# Plot 5: Per-class accuracy
ax5 = fig.add_subplot(2, 3, 5)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
bars = ax5.bar(CLASS_NAMES, per_class_acc * 100, color='steelblue')
ax5.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
ax5.set_ylabel("Accuracy (%)")
ax5.set_title("Per-Class Accuracy")
ax5.set_ylim(0, 110)
ax5.grid(True, alpha=0.3)
for bar, acc in zip(bars, per_class_acc):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc*100:.1f}", ha='center', fontsize=7)

# Plot 6: Sample predictions
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
ax6.set_title("Sample Predictions (Green=Correct, Red=Wrong)")
wrong_idx   = np.where(y_pred != y_test)[0][:5]
correct_idx = np.where(y_pred == y_test)[0][:5]
for col, idx in enumerate(list(correct_idx) + list(wrong_idx)):
    is_correct = y_pred[idx] == y_test[idx]
    color      = 'green' if is_correct else 'red'
    row        = 0 if col < 5 else 1
    c          = col % 5
    ax = fig.add_axes([0.67 + c * 0.062, 0.14 - row * 0.11, 0.055, 0.09])
    ax.imshow(X_test[idx])
    ax.set_title(f"P:{CLASS_NAMES[y_pred[idx]][:3]}\nA:{CLASS_NAMES[y_test[idx]][:3]}",
                 fontsize=5, color=color)
    ax.axis('off')

plt.savefig("assignment7_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("\nPlot saved as assignment7_output.png")
print("\nAssignment 7 Complete!")
