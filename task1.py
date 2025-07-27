import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, title=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 1000 == 0:
            avg = running_loss / 1000
            print(f"Training - Batch {i+1}/{len(dataloader)}, Avg Loss: {avg:.3f}")
            running_loss = 0.0


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def evaluate(model, dataloader, classes, device, print_every=100):
    model.eval()
    correct = 0
    total = 0
    correct_pred = dict.fromkeys(classes, 0)
    total_pred = dict.fromkeys(classes, 0)

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Update counts
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)
            correct += batch_correct
            total += batch_total

            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            # Print intermediate stats
            if (i + 1) % print_every == 0:
                interim_acc = 100 * correct / total
                print(f"Eval - Batch {i+1}/{len(dataloader)}, Interim Accuracy: {interim_acc:.2f}%")

    # Final overall accuracy
    overall_acc = 100 * correct / total
    print(f"\nOverall Accuracy: {overall_acc:.2f}% ({correct}/{total})")

    # Per-class accuracies
    print("\nPer-class Accuracy:")
    for cls in classes:
        cls_correct = correct_pred[cls]
        cls_total = total_pred[cls]
        cls_acc = 100 * cls_correct / cls_total if cls_total > 0 else 0.0
        print(f"  {cls:>5s}: {cls_acc:.2f}% ({cls_correct}/{cls_total})")

    # Return metrics if needed
    return {
        'overall_accuracy': overall_acc,
        'per_class_accuracy': {cls: 100 * correct_pred[cls] / total_pred[cls] if total_pred[cls] > 0 else 0.0 for cls in classes}
    }

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Show some training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images), title='Train Samples: ' + ' '.join(classes[labels[j]] for j in range(4)))

    # Train
    for epoch in range(2):
        print(f"\n==== Epoch {epoch+1} ====")
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        save_checkpoint(model, f"./cifar_net_epoch{epoch+1}.pth")

    # Evaluate
    metrics = evaluate(model, test_loader, classes, device, print_every=200)

    # Test on first batch
    images, labels = next(iter(test_loader))
    imshow(torchvision.utils.make_grid(images), title='GroundTruth: ' + ' '.join(classes[labels[j]] for j in range(4)))
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(classes[pred] for pred in preds.cpu()))


if __name__ == "__main__":
    main()
