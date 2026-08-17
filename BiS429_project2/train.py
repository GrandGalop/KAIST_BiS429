"""Train the predictive coding network on MNIST.

Example::

    python train.py --epochs 100 --batch-size 128 --lr 1e-3
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import data as dataset
from config import default_config
from model import NetworkForPredictiveCoding


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, help="default 100")
    parser.add_argument("--batch-size", type=int, help="default 128")
    parser.add_argument("--lr", type=float, help="default 1e-3")
    parser.add_argument("--layers", type=int, nargs="+",
                        help="layer sizes, default 784 500 500 10")
    parser.add_argument("--max-iterations", type=int,
                        help="inference relaxation steps per batch, default 50")
    parser.add_argument("--data-size", type=int,
                        help="truncate the training set for a quick smoke run")
    parser.add_argument("--data-root", default="MNIST",
                        help="where torchvision caches MNIST")
    parser.add_argument("--output-dir", default="results", type=Path)
    parser.add_argument("--device", help="cuda or cpu; autodetected by default")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def build_config(args):
    """Overlay the command line arguments onto the defaults."""
    cf = default_config()
    if args.epochs is not None:
        cf.n_epochs = args.epochs
    if args.batch_size is not None:
        cf.size_of_batch = args.batch_size
    if args.lr is not None:
        cf.lr = args.lr
    if args.max_iterations is not None:
        cf.max_iterations = args.max_iterations
    if args.data_size is not None:
        cf.data_size = args.data_size
    if args.seed is not None:
        cf.seed = args.seed
    if args.device is not None:
        cf.device = torch.device(args.device)
    if args.layers is not None:
        cf.numperceptrons = args.layers
        cf.numlayers = len(args.layers)
        cf.variance = torch.ones(cf.numlayers)
    return cf


def load_data(cf, data_root):
    """Load, truncate and preprocess both MNIST splits."""
    print("loading MNIST data...")
    train_set = dataset.load_mnist(data_root, train=True)
    test_set = dataset.load_mnist(data_root, train=False)
    img_train, label_train = dataset.get_imgs(train_set), dataset.get_labels(train_set)
    img_test, label_test = dataset.get_imgs(test_set), dataset.get_labels(test_set)

    if cf.data_size is not None:
        test_size = cf.data_size // 5
        img_train, label_train = img_train[:, : cf.data_size], label_train[:, : cf.data_size]
        img_test, label_test = img_test[:, :test_size], label_test[:, :test_size]

    print(f"img_train {img_train.shape} img_test {img_test.shape} "
          f"label_train {label_train.shape} label_test {label_test.shape}")

    print("performing preprocessing...")
    img_train, label_train = dataset.preprocess(img_train, label_train, cf)
    img_test, label_test = dataset.preprocess(img_test, label_test, cf)
    return img_train, label_train, img_test, label_test


def plot_accuracy(average_accuracy, output_dir):
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(len(average_accuracy)), average_accuracy)
    plt.title("Accuracy per Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    path = output_dir / "accuracy_per_epoch.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def main():
    args = parse_args()
    cf = build_config(args)
    print(f"device [{cf.device}]")

    if cf.seed is not None:
        np.random.seed(cf.seed)
        torch.manual_seed(cf.seed)

    img_train, label_train, img_test, label_test = load_data(cf, args.data_root)
    model = NetworkForPredictiveCoding(cf)
    average_accuracy = []

    # Gradients are derived by hand, so autograd is never needed.
    with torch.no_grad():
        for epoch in range(cf.n_epochs):
            print(f"\nepoch {epoch}")

            x_batches, y_batches = dataset.get_batches(img_train, label_train, cf.size_of_batch)
            print(f"training on {len(x_batches)} batches of size {cf.size_of_batch}")
            model.epoch_for_train(x_batches, y_batches, number_epoch=epoch)

            x_batches, y_batches = dataset.get_batches(img_test, label_test, cf.size_of_batch)
            print(f"testing on {len(x_batches)} batches of size {cf.size_of_batch}")
            accuracy_sets = model.epoch_for_test(x_batches, y_batches)

            mean_accuracy = float(np.mean(accuracy_sets))
            average_accuracy.append(mean_accuracy)
            print(f"average accuracy {mean_accuracy}")

            img_train, label_train = dataset.shuffle_columns(img_train, label_train)

    if not args.no_plot:
        plot_accuracy(average_accuracy, args.output_dir)


if __name__ == "__main__":
    main()
