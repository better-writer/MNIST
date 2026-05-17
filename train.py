import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 1x28x28 -> 16x14x14
        x = self.pool(self.relu(self.conv2(x)))   # 16x14x14 -> 32x7x7
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NormalizedCNN(nn.Module):
    """Wraps CNN with built-in MNIST normalization so ONNX accepts raw [0,1] input."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean = 0.1307
        self.std = 0.3081

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


def export_onnx(model, device):
    model.eval()
    # Move to CPU for ONNX export (ONNX does not carry GPU ops)
    cpu_model = model.cpu()
    wrapped = NormalizedCNN(cpu_model)
    wrapped.eval()
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        wrapped, dummy, "models/mnist_cnn.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print("Exported model to models/mnist_cnn.onnx")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    ort_session = ort.InferenceSession("models/mnist_cnn.onnx", providers=["CPUExecutionProvider"])
    result = ort_session.run(None, {"input": dummy.numpy()})
    pt_out = wrapped(dummy).detach().numpy()
    diff = np.abs(result[0] - pt_out).max()
    print(f"ONNX export verified — max diff from PyTorch: {diff:.6f}")

    # Move model back to original device for any further use
    model.to(device)


def plot_metrics(history, save_dir="outputs"):
    """Plot and save training curves and confusion matrix."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss & Accuracy curves side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Metrics", fontsize=14)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, history["test_acc"], "r-s", label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curve")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(f"{save_dir}/loss_accuracy.png", dpi=150)
    print(f"Saved loss/accuracy plot to {save_dir}/loss_accuracy.png")
    plt.close(fig)

    # 2. Per-class accuracy bar chart
    class_names = [str(i) for i in range(10)]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, history["per_class_acc"], color="skyblue", edgecolor="navy")
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, history["per_class_acc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/per_class_accuracy.png", dpi=150)
    print(f"Saved per-class accuracy plot to {save_dir}/per_class_accuracy.png")
    plt.close(fig)

    # 3. Confusion matrix heatmap
    cm = history["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    fig.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {save_dir}/confusion_matrix.png")
    plt.close(fig)

    # 4. Metrics summary table as text
    with open(f"{save_dir}/metrics_summary.txt", "w") as f:
        f.write("MNIST CNN Training Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Epochs: {len(history['train_loss'])}\n")
        f.write(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%\n")
        f.write(f"Best Test Accuracy: {max(history['test_acc']):.2f}%\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n\n")
        f.write("Epoch | Train Loss | Test Acc (%)\n")
        f.write("-" * 35 + "\n")
        for i, (loss, acc) in enumerate(zip(history["train_loss"], history["test_acc"]), 1):
            f.write(f"{i:5d} | {loss:.4f}    | {acc:.2f}\n")
        f.write("\nPer-Class Accuracy:\n")
        for cls, acc in enumerate(history["per_class_acc"]):
            f.write(f"  Digit {cls}: {acc:.2f}%\n")
    print(f"Saved metrics summary to {save_dir}/metrics_summary.txt")


def compute_per_class_accuracy(model, loader, device):
    """Compute per-class accuracy and confusion matrix."""
    class_correct = [0] * 10
    class_total = [0] * 10
    cm = np.zeros((10, 10), dtype=int)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for true, pred in zip(labels.cpu(), predicted.cpu()):
                class_total[true.item()] += 1
                class_correct[true.item()] += (true == pred).item()
                cm[true.item(), pred.item()] += 1

    per_class_acc = [
        100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(10)
    ]
    return per_class_acc, cm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {
        "train_loss": [],
        "test_acc": [],
    }

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {acc:.2f}%")

    # Compute detailed metrics on test set
    per_class_acc, cm = compute_per_class_accuracy(model, test_loader, device)
    history["per_class_acc"] = per_class_acc
    history["confusion_matrix"] = cm

    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    for cls, acc in enumerate(per_class_acc):
        print(f"  Digit {cls}: {acc:.2f}%")

    # Plot and save all metrics
    plot_metrics(history)

    # Save PyTorch model
    torch.save(model.state_dict(), "models/mnist_cnn.pth")
    print("Saved PyTorch model to models/mnist_cnn.pth")

    # Export to ONNX for C inference
    export_onnx(model, device)

if __name__ == "__main__":
    train()
