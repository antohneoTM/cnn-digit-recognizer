import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def load_dataset(batch_size):
    """Load MNIST dataset and return the train and test loaders"""

    # Defines transformation for data images
    transform = torchvision.transforms.Compose(
        [
            # Default MNIST transforms
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Datasets created and transformed into an universal format
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Train loaders for test and train datasets
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return (train_loader, test_loader)


def load_train_dataset(batch_size):
    """Load MNIST for train dataset and returns it"""

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=".data", train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader


def load_test_dataset(batch_size):
    """Load MNIST for test dataset and returns it"""

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_dataset = datasets.MNIST(
        root=".data", train=False, download=True, transform=transform
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    return test_loader
