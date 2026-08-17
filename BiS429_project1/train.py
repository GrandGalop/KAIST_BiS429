"""Train the chest X-ray classifier and write out test-set predictions.

Example::

    python train.py --data-dir data --epochs 100 --batch-size 30 --lr 0.00015
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.optim as optim

import data as dataset
from model import ChestXrayCNN, cross_entropy, n_correct

LABEL_NAMES = {0: "normal", 1: "abnormal"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", type=Path,
                        help="directory holding training_images/ and test_images/")
    parser.add_argument("--output-dir", default="results", type=Path,
                        help="where predictions and learning curves are written")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=30, type=int)
    parser.add_argument("--lr", default=0.00015, type=float)
    parser.add_argument("--lr-decay", default=0.9, type=float,
                        help="factor applied when validation accuracy plateaus")
    parser.add_argument("--n-train", default=360, type=int,
                        help="training samples; the remaining 800-n go to validation")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip writing the learning-curve figure")
    return parser.parse_args()


def evaluate(net, images, labels, batch_size, device):
    """Return ``(mean loss, accuracy)`` over the given split, without gradients."""
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in dataset.iter_batches(images, labels, batch_size, shuffle=False):
            x, y = x.to(device), y.to(device)
            output = net(x)
            total_loss += cross_entropy(output, y).item() * len(x)
            correct += n_correct(output, y)
    return total_loss / len(images), correct / len(images)


def train(net, optimizer, splits, args, generator):
    """Run the training loop, returning per-epoch loss and accuracy histories."""
    train_x, train_y, val_x, val_y = splits
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        net.train()
        for x, y in dataset.iter_batches(train_x, train_y, args.batch_size,
                                         generator=generator):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            cross_entropy(net(x), y).backward()
            optimizer.step()

        net.eval()
        train_loss, train_acc = evaluate(net, train_x, train_y, args.batch_size, args.device)
        val_loss, val_acc = evaluate(net, val_x, val_y, args.batch_size, args.device)
        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[key].append(value)

        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:6.2%} "
              f"| val loss {val_loss:.4f} acc {val_acc:6.2%}")

        # Decay the learning rate once validation accuracy stops moving.
        if epoch >= 10 and abs(history["val_acc"][-1] - history["val_acc"][-2]) <= 0.001:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay

        # Stop once the model is accurate and the train/val gap has closed.
        if train_acc >= 0.95 and abs(train_acc - val_acc) <= 0.03:
            print(f"early stop at epoch {epoch}")
            break

    return history


def write_predictions(net, test_images, path, device):
    """Write the answer sheet: one row per test image, 0 = normal, 1 = abnormal."""
    net.eval()
    with torch.no_grad():
        predicted = net(test_images.to(device)).argmax(dim=1).tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_image", "label", "prediction"])
        for i, label in enumerate(predicted, start=1):
            writer.writerow([i, label, LABEL_NAMES[label]])
    print(f"wrote {len(predicted)} predictions to {path}")
    return predicted


def plot_history(history, output_dir):
    """Save the loss/accuracy curves and the validation-minus-training loss gap."""
    from matplotlib import pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    _, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, title in zip(axes, ("loss", "acc"), ("Loss", "Accuracy")):
        ax.plot(history[f"train_{metric}"], label="train")
        ax.plot(history[f"val_{metric}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.savefig(output_dir / "learning_curves.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    gap = [v - t for v, t in zip(history["val_loss"], history["train_loss"])]
    plt.plot(gap)
    plt.xlabel("Epoch")
    plt.ylabel("Gap of Loss")
    plt.title("Validation - Training Loss over Epochs")
    plt.savefig(output_dir / "loss_gap.png", bbox_inches="tight")
    plt.close()
    print(f"wrote learning curves to {output_dir}")


def main():
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    images, labels = dataset.load_labeled_set(args.data_dir)
    splits = dataset.train_val_split(images, labels, args.n_train, generator=generator)
    test_images = dataset.load_test_set(args.data_dir)
    print(f"device {args.device} | train {len(splits[0])} | val {len(splits[2])} "
          f"| test {len(test_images)}")

    net = ChestXrayCNN().to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    history = train(net, optimizer, splits, args, generator)
    write_predictions(net, test_images, args.output_dir / "predictions.csv", args.device)
    if not args.no_plot:
        plot_history(history, args.output_dir)


if __name__ == "__main__":
    main()
