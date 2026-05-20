# Assignment 6: Multiclass Classifier using CNN
# Dataset: MNIST (keras built-in)
# Run: python assignment6_cnn_mnist.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Assignment 6: CNN Multiclass Classifier (MNIST)")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# ─────────────────────────────────────────────
# 1. Load and Preprocess Data
# ─────────────────────────────────────────────
print("\n[1] Loading MNIST Dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print(f"Image shape      : {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"Classes          : 0-9 (digits)")

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Reshape to add channel dimension (for CNN)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_ohe = keras.utils.to_categorical(y_train, 10)
y_test_ohe  = keras.utils.to_categorical(y_test,  10)

print(f"\nAfter preprocessing:")
print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"Pixel range   : [{X_train.min()}, {X_train.max()}]")

# ─────────────────────────────────────────────
# 2. Define CNN Model
# ─────────────────────────────────────────────
print("\n[2] Building CNN Model...")

model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
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

    # Fully Connected
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name="CNN_MNIST")

model.summary()

# ─────────────────────────────────────────────
# 3. Compile and Train
# ─────────────────────────────────────────────
print("\n[3] Compiling Model...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[4] Training Model (5 epochs)...")
history = model.fit(
    X_train, y_train_ohe,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ─────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────
print("\n[5] Evaluating Model...")
train_loss, train_acc = model.evaluate(X_train, y_train_ohe, verbose=0)
test_loss,  test_acc  = model.evaluate(X_test,  y_test_ohe,  verbose=0)

print(f"\nTraining  — Loss: {train_loss:.4f} | Accuracy: {train_acc*100:.2f}%")
print(f"Testing   — Loss: {test_loss:.4f}  | Accuracy: {test_acc*100:.2f}%")

# Predictions
y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# ─────────────────────────────────────────────
# 5. Visualizations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Assignment 6: CNN Multiclass Classifier (MNIST)", fontsize=14, fontweight='bold')

# Plot 1: Training Accuracy
axes[0, 0].plot(history.history['accuracy'],     label='Train', color='steelblue')
axes[0, 0].plot(history.history['val_accuracy'], label='Val',   color='tomato')
axes[0, 0].set_title("Model Accuracy")
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training Loss
axes[0, 1].plot(history.history['loss'],     label='Train', color='steelblue')
axes[0, 1].plot(history.history['val_loss'], label='Val',   color='tomato')
axes[0, 1].set_title("Model Loss")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[0, 2].imshow(cm, cmap='Blues')
axes[0, 2].set_title("Confusion Matrix")
axes[0, 2].set_xlabel("Predicted"); axes[0, 2].set_ylabel("Actual")
plt.colorbar(im, ax=axes[0, 2])
for i in range(10):
    for j in range(10):
        axes[0, 2].text(j, i, cm[i, j], ha='center', va='center',
                        fontsize=6, color='white' if cm[i, j] > cm.max()/2 else 'black')

# Plot 4: Sample predictions (correct)
correct_idx = np.where(y_pred == y_test)[0][:8]
axes[1, 0].set_title("Sample Correct Predictions")
axes[1, 0].axis('off')
for idx, sample in enumerate(correct_idx[:8]):
    ax = fig.add_axes([0.01 + idx*0.115, 0.08, 0.1, 0.12])
    ax.imshow(X_test[sample].reshape(28, 28), cmap='gray')
    ax.set_title(f"✓{y_pred[sample]}", fontsize=8, color='green')
    ax.axis('off')

# Plot 5: Sample predictions (wrong)
wrong_idx = np.where(y_pred != y_test)[0][:8]
axes[1, 1].set_title("Sample Wrong Predictions")
axes[1, 1].axis('off')
for idx, sample in enumerate(wrong_idx[:8]):
    ax = fig.add_axes([0.37 + idx*0.073, 0.08, 0.065, 0.1])
    ax.imshow(X_test[sample].reshape(28, 28), cmap='gray')
    ax.set_title(f"P:{y_pred[sample]}\nA:{y_test[sample]}", fontsize=6, color='red')
    ax.axis('off')

# Plot 6: Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
axes[1, 2].bar(range(10), per_class_acc * 100, color='steelblue')
axes[1, 2].set_xticks(range(10))
axes[1, 2].set_xlabel("Digit Class")
axes[1, 2].set_ylabel("Accuracy (%)")
axes[1, 2].set_title("Per-Class Accuracy")
axes[1, 2].set_ylim(90, 101)
axes[1, 2].grid(True, alpha=0.3)
for i, acc in enumerate(per_class_acc):
    axes[1, 2].text(i, acc*100 + 0.1, f"{acc*100:.1f}", ha='center', fontsize=7)

plt.tight_layout()
plt.savefig("assignment6_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("\nPlot saved as assignment6_output.png")
print("\nAssignment 6 Complete!")
