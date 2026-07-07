from preprocess import preprocess_data

X_train, y_train, X_test, y_test = preprocess_data()

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print("\nPixel Range")
print("Minimum :", X_train.min())
print("Maximum :", X_train.max())