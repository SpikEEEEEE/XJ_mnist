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

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def train_step(self, x, y, optimizer):
        self.train()
        optimizer.zero_grad()
        y_pred = self(x)
        loss = nn.CrossEntropyLoss()(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, loader):
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():
            for x, y in loader:
                logits = self(x)
                total_loss += loss_fn(logits, y).item()
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_samples += y.size(0)

        return total_loss / total_samples, total_correct / total_samples
        

def main():
    transform = transforms.ToTensor()

    fashion_mnist = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        fashion_mnist,
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        fashion_mnist,
        batch_size=64,
        shuffle=True,
    )

    model = CNN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        
        for x, y in train_loader:
            loss = model.train_step(x, y, optimizer)
            
            total_train_loss += loss * y.size(0)
            total_train_samples += y.size(0)

        train_loss = total_train_loss / total_train_samples

        test_loss, test_acc = model.evaluate(x, y)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.plot(test_accs, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()