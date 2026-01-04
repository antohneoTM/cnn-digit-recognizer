import PIL
import torch
import torchvision

import os

from cnn_model import Model

from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent.parent / "model" / "digit-recognizer.pth"
SAMPLE_IMG_PATH = Path(__file__).parent.parent / "sample_images"


def process_image(image_path):
    image = PIL.Image.open(image_path)

    transform = torchvision.transforms.Compose(
        [
            # MNIST is grayscale and has a default size of 28 x 28
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    return transform(image).unsqueeze(0)


def predict_image(image_path, model_path, device):
    """Predict digit of single image with trained model"""

    # Load model into an evaluation state
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_tensor = process_image(image_path)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"Predicted Digit: {predicted_class}")


if __name__ == "__main__":
    # Load model into an evaluation state
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Search through every directory in 'sample_images' directory
    # and predict the image within each directory
    for root, dirs, files in os.walk(SAMPLE_IMG_PATH):
        print("Predicting images within:", root)

        for file in files:
            print(f"Predicting file: {file}")

            input_tensor = process_image(root+"/"+file)

            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
            print(f"Predicted Digit: {predicted_class}")
        print("")
