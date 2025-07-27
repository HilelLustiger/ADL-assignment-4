import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Utility to show images
def imshow(img, title=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Model with encoder, classification head, and decoder
class Maya(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)
        # Decoder layers
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5)
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(6, 3, 5)
        # Classification head
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1(x))            # [N,6,28,28]
        self.size1, _ = x1.size(), None
        x1_p, self.idx1 = self.pool1(x1)      # [N,6,14,14]

        x2 = F.relu(self.conv2(x1_p))         # [N,16,10,10]
        self.size2 = x2.size()
        x2_p, self.idx2 = self.pool2(x2)      # [N,16,5,5]

        # Classification head
        flat = x2_p.view(x2_p.size(0), -1)
        y = F.relu(self.fc1(flat))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        # Decoder
        d2 = self.unpool2(x2_p, self.idx2, output_size=self.size2)
        d2 = F.relu(self.deconv2(d2))
        d1 = self.unpool1(d2, self.idx1, output_size=self.size1)
        d1 = F.relu(d1)
        x_recon = torch.tanh(self.deconv1(d1))

        return y, x_recon

# Training and evaluation functions
def train_one_epoch(model, dataloader, optimizer, cls_criterion, rec_criterion, device, lambda_rec):
    model.train()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, recon = model(inputs)
        loss = cls_criterion(outputs, labels) + lambda_rec * rec_criterion(recon, inputs)
        loss.backward()
        optimizer.step()
        if (i+1) % 2500 == 0:
            print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.3f}")

def evaluate(model, dataloader, classes, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    acc = 100 * correct / total
    print(f"Overall accuracy: {acc:.2f}%")
    return acc

# Latent channel analysis
def channel_analysis(model, img, layer, channels, device):

    model.eval()
    img = img.to(device)
    with torch.no_grad():
        # forward to set indices
        model(img)
        # recompute feature maps
        f1 = F.relu(model.conv1(img))            # [1,6,28,28]
        p1, idx1 = model.pool1(f1)              # [1,6,14,14]
        f2 = F.relu(model.conv2(p1))             # [1,16,10,10]
        p2, idx2 = model.pool2(f2)              # [1,16,5,5]

    recons = []
    if layer == 'conv1':
        base = p1
        idx = idx1
        size = f1.size()
        for ch in channels:
            z = torch.zeros_like(base)
            z[:,ch] = base[:,ch]
            u = model.unpool1(z, idx, output_size=size)
            u = F.relu(u)
            rec = torch.tanh(model.deconv1(u))
            recons.append(rec.cpu().squeeze(0))
    else:
        base = p2
        idx = idx2
        size = f2.size()
        for ch in channels:
            z = torch.zeros_like(base)
            z[:,ch] = base[:,ch]
            u2 = model.unpool2(z, idx, output_size=size)
            u2 = F.relu(model.deconv2(u2))
            u1 = model.unpool1(u2, idx1, output_size=f1.size())
            u1 = F.relu(u1)
            rec = torch.tanh(model.deconv1(u1))
            recons.append(rec.cpu().squeeze(0))
    return recons

# Main script
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    train_set = torchvision.datasets.CIFAR10('./data', True, transform, download=True)
    test_set  = torchvision.datasets.CIFAR10('./data', False, transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=4, shuffle=False, num_workers=2)
    classes = train_set.classes

    model = Maya().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    cls_criterion = nn.CrossEntropyLoss()
    rec_criterion = nn.MSELoss()
    lambda_rec = 1.0

    # Train for 2 epochs
    for epoch in range(2):
        print(f"Epoch {epoch+1}")
        train_one_epoch(model, train_loader, optimizer, cls_criterion, rec_criterion, device, lambda_rec)
        torch.save(model.state_dict(), f"maya_epoch{epoch+1}.pth")

    # Evaluate
    evaluate(model, test_loader, classes, device)

    # Latent analysis for two images
    choices = {'conv1': list(range(6)), 'conv2': [0,8,15]}
    for name, loader in [('Train', train_loader), ('Test', test_loader)]:
        img, _ = next(iter(loader))
        img = img[0:1]
        print(f"Latent analysis on {name} image")
        orig = img.cpu().squeeze(0) / 2 + 0.5
                # Plot grid with original last
        fig, axes = plt.subplots(4, 3, figsize=(9, 12))
        axes = axes.flatten()
        idx = 0
        # conv1 channels
        for ch in choices['conv1']:
            rec = channel_analysis(model, img, 'conv1', [ch], device)[0]
            rec_vis = (rec / 2 + 0.5).permute(1,2,0).detach().numpy()
            axes[idx].imshow(rec_vis)
            axes[idx].set_title(f'z1 ch{ch}')
            axes[idx].axis('off')
            idx += 1
        # conv2 channels
        for ch in choices['conv2']:
            rec = channel_analysis(model, img, 'conv2', [ch], device)[0]
            rec_vis = (rec / 2 + 0.5).permute(1,2,0).detach().numpy()
            axes[idx].imshow(rec_vis)
            axes[idx].set_title(f'z2 ch{ch}')
            axes[idx].axis('off')
            idx += 1
        # original last
        orig_vis = orig.permute(1,2,0).detach().numpy()
        axes[idx].imshow(orig_vis)
        axes[idx].set_title('Original')
        axes[idx].axis('off')
        idx += 1
        # turn off remaining
        for j in range(idx, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()