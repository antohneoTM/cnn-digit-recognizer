import torch
import load_dataset
from train_cnn import train_cnn
from evaluate_cnn import evaluate_cnn

from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent.parent / "model" / "digit-recognizer.pth"


def run() -> None:
    train_loader = load_dataset.load_train_dataset(batch_size=256)
    test_loader = load_dataset.load_test_dataset(batch_size=60000)

    while True:
        user_input = input("[T]rain model, [E]valuate model or [Q]uit\n")
        if user_input == "t" or user_input == "T":
            train_cnn(train_loader, MODEL_PATH, DEVICE)

        elif user_input == "e" or user_input == "E":
            try:
                evaluate_cnn(test_loader, MODEL_PATH, DEVICE)
            except FileNotFoundError:
                print(f"Error: Model file not found '{MODEL_PATH}'")
                continue
            except OSError:
                print("Error: An error occured opening or processing model")
                continue
            except Exception as e:
                print(f"An unexpected error occured: {e}")

        elif user_input == "q" or user_input == "Q":
            break

        else:
            print("Please enter a proper option:\n")


if __name__ == "__main__":
    run()
