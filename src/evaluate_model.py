"""
evaluate_model.py

This file evaluates the trained CNN model.
"""

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocess import preprocess_data


def evaluate_model(history=None):

    # Load processed dataset
    X_train, y_train, X_test, y_test = preprocess_data()

    # Load saved model
    model = load_model("models/cnn_mnist.keras")

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest Accuracy : {accuracy*100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    # If history is available, save graphs
    if history is not None:

        # Accuracy graph
        plt.figure(figsize=(8,5))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig("output/accuracy.png")
        plt.close()

        # Loss graph
        plt.figure(figsize=(8,5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("output/loss.png")
        plt.close()

        print("\nAccuracy and Loss graphs saved!")

    # Predictions
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predicted_labels)

    plt.figure(figsize=(8,8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=range(10)
    )

    disp.plot(cmap="Blues")

    plt.savefig("output/confusion_matrix.png")
    plt.close()

    print("Confusion Matrix saved!")

    return accuracy