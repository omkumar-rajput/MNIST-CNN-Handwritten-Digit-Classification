from train_model import train_model
from predict_image import predict_uploaded_image
from visualize_dataset import visualize_samples
def display_menu():
    print("\n" + "=" * 50)
    print(" CNN Model on MNIST Dataset")
    print(" Handwritten Digit Classification")
    print("=" * 50)

    print("1. Dataset Information")
    print("2. View Sample Digits")
    print("3. Train CNN Model")
    print("4. Evaluate Model")
    print("5. Predict Uploaded Image")
    print("6. Exit")


def main():
    while True:

        display_menu()

        choice = input("\nEnter your choice: ")

        if choice == "1":
            print("\nDataset Information feature coming soon...\n")

        elif choice == "2":
            visualize_samples()

        elif choice == "3":
            history, model = train_model()

        elif choice == "4":
            from evaluate_model import evaluate_model
            evaluate_model()

        elif choice == "5":
            predict_uploaded_image()

        elif choice == "6":
            print("\nThank you for using the project!")
            break

        else:
            print("\nInvalid Choice! Please try again.\n")


if __name__ == "__main__":
    main()