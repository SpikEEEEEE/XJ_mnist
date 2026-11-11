import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def train_step(self, x, y, optimizer):
        self.train()
        optimizer.zero_grad()
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, loader):
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for x, y in loader:
                logits = self(x)
                batch_size = y.size(0)
                total_loss += self.loss_fn(logits, y).item() * batch_size
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += batch_size

        return total_loss / total_samples, total_correct / total_samples
        

def main():
    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.FashionMNIST(
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

    model = CNN()
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

        test_loss, test_acc = model.evaluate(test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Acc: {test_acc:.4f}"
        )

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.plot(test_accs, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
