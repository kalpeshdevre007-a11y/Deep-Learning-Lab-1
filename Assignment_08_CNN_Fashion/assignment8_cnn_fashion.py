# Assignment 8: Fashion Clothing Classifier using CNN
# Dataset: Fashion-MNIST (keras built-in)
# Run: python assignment8_cnn_fashion.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Assignment 8: Fashion Clothing Classifier (CNN)")
print("  Dataset: Fashion-MNIST")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# ─────────────────────────────────────────────
# 1. Load and Preprocess Data
# ─────────────────────────────────────────────
print("\n[1] Loading Fashion-MNIST Dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print(f"Image shape      : {X_train.shape[1]}x{X_train.shape[2]} (Grayscale)")
print(f"Classes          : {CLASS_NAMES}")

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Reshape to add channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_ohe = keras.utils.to_categorical(y_train, 10)
y_test_ohe  = keras.utils.to_categorical(y_test,  10)

print(f"\nAfter preprocessing:")
print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"Pixel range   : [{X_train.min():.1f}, {X_train.max():.1f}]")

# Class distribution
print("\nClass Distribution (Training):")
for i, name in enumerate(CLASS_NAMES):
    count = np.sum(y_train == i)
    bar = "█" * (count // 500)
    print(f"  {i}: {name:<15} {count}  {bar}")

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

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    # Fully Connected
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name="CNN_FashionMNIST")

model.summary()

# ─────────────────────────────────────────────
# 3. Compile and Train
# ─────────────────────────────────────────────
print("\n[3] Compiling Model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6
)
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
)

print("\n[4] Training Model (15 epochs max, early stopping enabled)...")
history = model.fit(
    X_train, y_train_ohe,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=[lr_scheduler, early_stop],
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

y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# 5. Visualizations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Assignment 8: Fashion Clothing Classifier (Fashion-MNIST CNN)", fontsize=14, fontweight='bold')

# Plot 1: Sample images from each class
axes[0, 0].set_title("Sample Images — All 10 Categories", fontsize=10)
axes[0, 0].axis('off')
grid = np.zeros((28 * 2, 28 * 5), dtype=np.float32)
for i in range(10):
    idx = np.where(y_test == i)[0][0]
    row, col = divmod(i, 5)
    grid[row*28:(row+1)*28, col*28:(col+1)*28] = X_test[idx].reshape(28, 28)
axes[0, 0].imshow(grid, cmap='gray')
for i, name in enumerate(CLASS_NAMES):
    row, col = divmod(i, 5)
    axes[0, 0].text(col*28 + 14, row*28 + 26, name[:7],
                    ha='center', va='bottom', fontsize=6, color='yellow')

# Plot 2: Accuracy curve
axes[0, 1].plot(history.history['accuracy'],     label='Train', color='steelblue', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Val',   color='tomato',    linewidth=2)
axes[0, 1].set_title("Model Accuracy per Epoch")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Loss curve
axes[0, 2].plot(history.history['loss'],     label='Train', color='steelblue', linewidth=2)
axes[0, 2].plot(history.history['val_loss'], label='Val',   color='tomato',    linewidth=2)
axes[0, 2].set_title("Model Loss per Epoch")
axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("Loss")
axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
short_names = [n[:6] for n in CLASS_NAMES]
im = axes[1, 0].imshow(cm, cmap='Blues')
axes[1, 0].set_xticks(range(10)); axes[1, 0].set_yticks(range(10))
axes[1, 0].set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
axes[1, 0].set_yticklabels(short_names, fontsize=7)
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Predicted"); axes[1, 0].set_ylabel("Actual")
for i in range(10):
    for j in range(10):
        axes[1, 0].text(j, i, cm[i, j], ha='center', va='center',
                        fontsize=6, color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[1, 0])

# Plot 5: Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
colors = ['#2ecc71' if a >= 0.90 else '#e67e22' if a >= 0.80 else '#e74c3c' for a in per_class_acc]
bars = axes[1, 1].bar(range(10), per_class_acc * 100, color=colors)
axes[1, 1].set_xticks(range(10))
axes[1, 1].set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
axes[1, 1].set_ylabel("Accuracy (%)")
axes[1, 1].set_title("Per-Class Accuracy (Green≥90%, Orange≥80%, Red<80%)")
axes[1, 1].set_ylim(0, 115)
axes[1, 1].grid(True, alpha=0.3)
for bar, acc in zip(bars, per_class_acc):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{acc*100:.1f}%", ha='center', fontsize=7)

# Plot 6: Correct vs Wrong predictions grid
axes[1, 2].set_title("Predictions: ✓ Correct (top) | ✗ Wrong (bottom)", fontsize=9)
axes[1, 2].axis('off')
correct_idx = np.where(y_pred == y_test)[0][:5]
wrong_idx   = np.where(y_pred != y_test)[0][:5]
for col, idx in enumerate(correct_idx):
    ax = fig.add_axes([0.675 + col * 0.063, 0.19, 0.055, 0.1])
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"✓{CLASS_NAMES[y_pred[idx]][:5]}", fontsize=5, color='green')
    ax.axis('off')
for col, idx in enumerate(wrong_idx):
    ax = fig.add_axes([0.675 + col * 0.063, 0.06, 0.055, 0.1])
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"P:{CLASS_NAMES[y_pred[idx]][:4]}\nA:{CLASS_NAMES[y_test[idx]][:4]}",
                 fontsize=5, color='red')
    ax.axis('off')

plt.savefig("assignment8_output.png", dpi=100, bbox_inches='tight')
plt.show()
print("\nPlot saved as assignment8_output.png")
print("\nAssignment 8 Complete!")
