"""
train_model.py

This file builds, trains and saves the CNN model.
"""

# Import TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Import preprocessing function
from preprocess import preprocess_data


def build_model():

    model = Sequential()

    # First Convolution Layer
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation="relu",
            input_shape=(28,28,1)
        )
    )

    # First Pooling Layer
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Second Convolution Layer
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation="relu"
        )
    )

    # Second Pooling Layer
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(128, activation="relu"))

    # Dropout
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(10, activation="softmax"))

    return model


def train_model():

    # Load processed data
    X_train, y_train, X_test, y_test = preprocess_data()

    # Build model
    model = build_model()

    print("\n==============================")
    print("CNN MODEL SUMMARY")
    print("==============================\n")

    model.summary()

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining Started...\n")

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Save Model
    model.save("models/cnn_mnist.keras")

    print("\nModel Saved Successfully!")

    return history, model