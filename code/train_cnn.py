# from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import Model


def train_cnn(train_loader, model_path, device):
    """Takes a loaded train dataset and trains CNN model with Model class.
    Saves model at MODEL_PATH when done. ~99.2 to 99.5% accurate."""

    learning_rate = 0.001  # Initial learning rate for first 20 epochs
    num_epochs = 30

    # Create model object
    model = Model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()  # Set model into training mode

        running_loss = 0.0  # Tracks loss of each epoch

        # When 20 epochs are reached lower learning rate for accurate training
        if epoch >= 20:
            learning_rate = 0.0001

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save model state dict at MODEL_PATH
    torch.save(model.state_dict(), model_path)
    print("Model has been successfully trained and saved...\n")
