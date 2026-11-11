import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

def evaluate(model, loader):
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            total_loss += model.loss(x, y).item()
            total_samples += x.size(0)
            total_correct += (model.predict(x) == y).sum().item()
    return total_loss / total_samples, total_correct / total_samples

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.linear3(x)  # 原始 logits，不加 Softmax

    def predict(self, x):
        return self(x).argmax(dim=1)

    def loss(self, x, y):
        return self.loss_fn(self(x), y)

    def accuracy(self, x, y):
        with torch.no_grad():
            return (self.predict(x) == y).float().mean().item()

    def train_step(self, x, y, optimizer):
        self.train()  # 调用 nn.Module 的训练模式
        optimizer.zero_grad()
        loss = self.loss(x, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test(self, x, y):
        self.eval()
        return self.accuracy(x, y)

def main():
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
    )

    images, labels = next(iter(train_loader))
    print(images.shape)  # torch.Size([64, 1, 28, 28])
    print(labels.shape)  # torch.Size([64])

    model = MNISTNet()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 20

    train_losses = []
    test_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for x, y in train_loader:
            loss = model.train_step(x, y, optimizer)

            total_train_loss += loss * y.size(0)
            total_train_samples += y.size(0)

        train_loss = total_train_loss / total_train_samples
        test_loss, test_acc = evaluate(model, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_model.pth")

    model.load_state_dict(
        torch.load("mnist_model.pth")
    )

    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Acc: {test_acc:.4f}")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.plot(test_accs, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

