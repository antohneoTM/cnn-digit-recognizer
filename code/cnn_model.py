import torch.nn as nn


class Model(nn.Module):
    """Model for convolutional neural network trained with MNIST dataset"""

    def __init__(self):
        super(Model, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten_layer1 = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=10)
        )

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = out.reshape(-1, 64*7*7)
        out = self.flatten_layer1(out)

        return out
