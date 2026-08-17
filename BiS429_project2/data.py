"""MNIST loading, preprocessing and batching for the predictive coding network.

Everything is kept in ``(feature, batch)`` orientation — images are
``(784, N)`` and one-hot labels are ``(10, N)`` — because the predictive
coding update rules are written as left-multiplications by the weights.
"""

import numpy as np
import torch
import torchvision

import functions as F

N_PIXELS = 784
N_CLASSES = 10


def as_tensor(array, device):
    """Move a numpy array onto ``device`` as a float32 tensor."""
    return torch.from_numpy(array).float().to(device)


def load_mnist(root="MNIST", train=True):
    """Download (once) and return the MNIST split."""
    return torchvision.datasets.MNIST(root, download=True, train=train)


def _onehot(label, n_classes=N_CLASSES):
    arr = np.zeros([n_classes])
    arr[int(label)] = 1.0
    return arr


def get_imgs(dataset):
    """Flatten every image to 784 pixels in [0, 1] and stack as ``(784, N)``."""
    imgs = np.array([np.array(img).reshape([N_PIXELS]) / 255.0 for img, _ in dataset])
    return np.swapaxes(imgs, 0, 1)


def get_labels(dataset):
    """Stack one-hot labels as ``(10, N)``."""
    labels = np.array([_onehot(label) for _, label in dataset])
    return np.swapaxes(labels, 0, 1)


def scale(array, factor):
    """Shrink values toward 0.5 so the inverse activation stays finite."""
    return array * factor + 0.5 * (1 - factor) * np.ones(array.shape)


def preprocess(imgs, labels, cf):
    """Apply the optional scaling and inverse-activation steps to one split."""
    if cf.apply_scaling:
        imgs = scale(imgs, cf.img_scale)
        labels = scale(labels, cf.label_scale)
    if cf.apply_inv:
        imgs = F.f_inv(imgs, cf.activation_function)
    return imgs, labels


def get_batches(imgs, labels, size_of_batch):
    """Split along the batch axis; the last batch may be smaller."""
    n_data = imgs.shape[1]
    starts = range(0, n_data, size_of_batch)
    img_batches = [imgs[:, s : s + size_of_batch] for s in starts]
    label_batches = [labels[:, s : s + size_of_batch] for s in starts]
    return img_batches, label_batches


def shuffle_columns(imgs, labels, rng=None):
    """Reshuffle the dataset between epochs, keeping images and labels aligned."""
    rng = rng or np.random
    perm = rng.permutation(imgs.shape[1])
    return imgs[:, perm], labels[:, perm]


def accuracy(pred_labels, labels):
    """Fraction of the batch whose argmax prediction matches the one-hot label."""
    correct = (torch.argmax(pred_labels, dim=0) == torch.argmax(labels, dim=0)).sum().item()
    return correct / pred_labels.size(1)
