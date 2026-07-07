"""
preprocess.py

This file is responsible for preparing the MNIST dataset
before it is used for training the CNN model.
"""

# Import NumPy
import numpy as np

# Import the function we created earlier
from load_dataset import load_mnist


def preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        X_train
        y_train
        X_test
        y_test
    """

    # Load dataset
    X_train, y_train, X_test, y_test = load_mnist()

    # Normalize pixel values
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape images
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, y_train, X_test, y_test