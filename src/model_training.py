import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer, metric, device):
    model.train()
    size = len(dataloader.dataset)
    last_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        last_loss = loss.item()  # Track the loss of the last batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric(pred, y)

        if batch % 20 == 0:
            print(f"Loss: {last_loss:.4f}  [{batch * len(X)}/{size}]")

    train_acc = metric.compute().item()
    print(f"Training Accuracy: {train_acc}, Loss: {last_loss:.4f}")
    metric.reset()
    return last_loss, train_acc

def test_loop(dataloader, model, metric, device):
    model.eval()
    last_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            last_loss = loss.item()  # Track the loss of the last batch
            metric(pred, y)

    test_acc = metric.compute().item()
    print(f"Testing Accuracy: {test_acc}, Loss: {last_loss:.4f}")
    metric.reset()
    return last_loss, test_acc

