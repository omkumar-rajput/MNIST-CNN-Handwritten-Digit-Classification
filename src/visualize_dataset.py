"""
visualize_dataset.py

This file is responsible for displaying and saving
sample images from the MNIST dataset.
"""

# Import matplotlib
import matplotlib.pyplot as plt

# Import the function we created
from load_dataset import load_mnist


def visualize_samples():

    # Load the dataset
    X_train, y_train, X_test, y_test = load_mnist()

    # Create a figure
    plt.figure(figsize=(8, 8))

    # Display the first 9 images
    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(X_train[i], cmap="gray")

        plt.title(f"Label : {y_train[i]}")

        plt.axis("off")

    # Save the figure
    plt.savefig("output/sample_digits.png")

    print("\n✅ Sample digits saved to output/sample_digits.png")