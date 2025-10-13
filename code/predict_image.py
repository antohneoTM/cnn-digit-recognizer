import PIL
import torch
import torchvision

from cnn_model import Model

from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent.parent / "model" / "digit-recognizer.pth"


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
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_tensor = process_image(image_path)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"Predicted Digit: {predicted_class}")


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    predict_image(image_path, MODEL_PATH, DEVICE)
