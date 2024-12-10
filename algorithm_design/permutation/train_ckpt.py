import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 64)
        self.linear2 = nn.Linear(64, 24)
        self.linear3 = nn.Linear(24, 10)

    def forward(self, image):
        x = image.view(-1, 784)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def main():
    use_cuda = torch.cuda.is_available()
    print(f"Use GPU: {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader = DataLoader(
        datasets.MNIST(
            r'D:\Dataset\MNIST',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )  # create dataloader
    model = Model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    for epoch in range(1, 21):
        train(model, device, train_loader, optimizer, epoch)
    save_model(model, "model_ckpt2.pth")


if __name__ == '__main__':
    main()
