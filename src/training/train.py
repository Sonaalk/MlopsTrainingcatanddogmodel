import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import mlflow

mlflow.set_experiment("Cats_vs_Dogs_CNN")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

def train():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder("data/processed/train", transform=transform)
    val_ds = datasets.ImageFolder("data/processed/val", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    with mlflow.start_run():
        for epoch in range(3):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

        model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                preds.extend(outputs.argmax(1).tolist())
                labels.extend(y.tolist())

        acc = accuracy_score(labels, preds)
        mlflow.log_metric("val_accuracy", acc)

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/cnn_model.pt")
        mlflow.log_artifact("models/cnn_model.pt")

        print(f"Validation Accuracy: {acc}")

if __name__ == "__main__":
    train()