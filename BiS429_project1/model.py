"""CNN classifier and loss for the chest X-ray binary diagnosis task."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChestXrayCNN(nn.Module):
    """Two conv blocks (32 then 4 channels) followed by three fully connected layers.

    ``forward`` returns class *probabilities*, not logits, so it pairs with
    :func:`cross_entropy` below rather than ``nn.CrossEntropyLoss``.
    """

    def __init__(self, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 4, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 -> 32
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


def cross_entropy(y_pred, y_true, eps=1e-9):
    """Cross entropy over one-hot targets and already-normalized probabilities."""
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


def n_correct(y_pred, y_true):
    """Number of samples whose argmax class matches the one-hot target."""
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).sum().item()
