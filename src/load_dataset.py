from tensorflow.keras.datasets import mnist


def load_mnist():
    """
    Load the MNIST dataset.

    Returns:
        X_train, y_train, X_test, y_test
    """

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    return X_train, y_train, X_test, y_test