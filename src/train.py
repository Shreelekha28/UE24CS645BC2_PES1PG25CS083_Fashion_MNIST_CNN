import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
sys.path.append("src")

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist

from convolution import Convolution
from pooling import MaxPool
from fully_connected import Softmax
from loss import cross_entropy_loss

# -----------------------------
# LOAD DATASET
# -----------------------------

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize images
x_train = x_train / 255
x_test = x_test / 255

print("Dataset Loaded Successfully!")

# -----------------------------
# CNN ARCHITECTURE
# -----------------------------

# Increased filters from 8 -> 16
conv = Convolution(16, 3)

pool = MaxPool()

# 13x13x16 after pooling
softmax = Softmax(13 * 13 * 16, 10)

# -----------------------------
# TRACKING METRICS
# -----------------------------

losses = []
accuracies = []

# -----------------------------
# FORWARD PASS
# -----------------------------

def forward(image, label):

    out = conv.forward(image)

    out = pool.forward(out)

    out = softmax.forward(out)

    loss = cross_entropy_loss(out, label)

    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

# -----------------------------
# TRAIN FUNCTION
# -----------------------------

def train(image, label, lr=0.001):

    out, loss, acc = forward(image, label)

    # Initial gradient
    gradient = np.zeros(10)

    gradient[label] = -1 / out[label]

    # Backpropagation
    softmax.backward(gradient, lr)

    return loss, acc

# -----------------------------
# TRAINING LOOP
# -----------------------------

epochs = 15

for epoch in range(epochs):

    print(f"\n========== Epoch {epoch + 1} ==========")

    loss = 0
    num_correct = 0

    # Increased training data: 5000 images
    for i, (image, label) in enumerate(
        zip(x_train[:5000], y_train[:5000])
    ):

        l, acc = train(image, label)

        loss += l

        num_correct += acc

        # Print every 500 steps
        if (i + 1) % 500 == 0:

            avg_loss = loss / 500

            avg_acc = (num_correct / 500) * 100

            print(
                f"[Step {i + 1}] "
                f"Loss: {avg_loss:.3f} | "
                f"Accuracy: {avg_acc:.2f}%"
            )

            losses.append(avg_loss)

            accuracies.append(avg_acc)

            loss = 0

            num_correct = 0

# -----------------------------
# SAVE ACCURACY GRAPH
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(accuracies)

plt.title("Accuracy Across Epochs")

plt.xlabel("Iterations")

plt.ylabel("Accuracy (%)")

plt.grid(True)

plt.savefig("graphs/accuracy.png")

# -----------------------------
# SAVE LOSS GRAPH
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(losses)

plt.title("Loss Across Epochs")

plt.xlabel("Iterations")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig("graphs/loss.png")

plt.show()

# -----------------------------
# TEST EVALUATION
# -----------------------------

print("\nEvaluating on Test Dataset...\n")

test_loss = 0
test_correct = 0

# Increased test samples: 2000
for image, label in zip(x_test[:2000], y_test[:2000]):

    out = conv.forward(image)

    out = pool.forward(out)

    out = softmax.forward(out)

    loss = cross_entropy_loss(out, label)

    test_loss += loss

    if np.argmax(out) == label:

        test_correct += 1

# Final metrics
avg_test_loss = test_loss / 2000

test_accuracy = (test_correct / 2000) * 100

print(f"Final Test Loss: {avg_test_loss:.3f}")

print(f"Final Test Accuracy: {test_accuracy:.2f}%")