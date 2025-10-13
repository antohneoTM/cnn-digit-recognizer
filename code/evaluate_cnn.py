import torch
from cnn_model import Model


def evaluate_cnn(test_loader, model_path, device):
    """Evaluates a fully trained model with the test_loader variable"""

    # Create model object
    model = Model()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()  # Set model into an evaluation mode
    correct_tests = total_tests = 0  # Tracks number of correct tests by model

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_tests += labels.size(0)
            correct_tests += (predicted == labels).sum().item()

    accuracy = 100 * correct_tests / total_tests
    print(f"Test Accuracy: {accuracy:.2f}%")
