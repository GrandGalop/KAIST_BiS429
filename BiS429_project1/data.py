"""Chest X-ray dataset loading, splitting and batching.

Images are 128x128 grayscale PNGs laid out as::

    <data-dir>/training_images/normal{1..400}.png
    <data-dir>/training_images/abnormal{1..400}.png
    <data-dir>/test_images/{1..50}.png

Every loader returns tensors shaped ``(N, 1, 128, 128)`` for images and
``(N, 2)`` one-hot ``long`` tensors for labels (``[1, 0]`` normal,
``[0, 1]`` abnormal).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

NORMAL = (1, 0)
ABNORMAL = (0, 1)

N_PER_CLASS = 400
N_TEST = 50


def _load_image(path):
    """Read one PNG as a normalized ``(1, H, W)`` float array."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Unzip Project1_data_files.zip into the data directory first."
        )
    image = np.array(Image.open(path), dtype=np.float32) / 255.0
    return image[np.newaxis, :, :]


def load_labeled_set(data_dir, n_per_class=N_PER_CLASS):
    """Load the 800 labeled training images (normal and abnormal interleaved)."""
    image_dir = Path(data_dir) / "training_images"
    images, labels = [], []
    for i in range(1, n_per_class + 1):
        images.append(_load_image(image_dir / f"normal{i}.png"))
        labels.append(NORMAL)
        images.append(_load_image(image_dir / f"abnormal{i}.png"))
        labels.append(ABNORMAL)
    return torch.from_numpy(np.stack(images)), torch.tensor(labels, dtype=torch.long)


def load_test_set(data_dir, n_test=N_TEST):
    """Load the 50 unlabeled test images, ordered 1..50 as the answer sheet expects."""
    image_dir = Path(data_dir) / "test_images"
    images = [_load_image(image_dir / f"{i}.png") for i in range(1, n_test + 1)]
    return torch.from_numpy(np.stack(images))


def train_val_split(images, labels, n_train=360, generator=None):
    """Shuffle once, then take the first ``n_train`` samples as the training split."""
    order = torch.randperm(len(images), generator=generator)
    train, val = order[:n_train], order[n_train:]
    return images[train], labels[train], images[val], labels[val]


def iter_batches(images, labels, batch_size, shuffle=True, generator=None):
    """Yield ``(x, y)`` batches; the final batch may be smaller than ``batch_size``."""
    order = (
        torch.randperm(len(images), generator=generator)
        if shuffle
        else torch.arange(len(images))
    )
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        yield images[index], labels[index]
