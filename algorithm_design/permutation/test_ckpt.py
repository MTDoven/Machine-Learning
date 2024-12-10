import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 64)
        self.linear2 = nn.Linear(64, 24)
        self.linear3 = nn.Linear(24, 10)

    def forward(self, image):
        image = F.avg_pool2d(image, (2, 2), 2)
        x = image.view(-1, 196)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')


def main():
    use_cuda = torch.cuda.is_available()
    print(f"Use GPU: {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")
    test_loader = DataLoader(
        datasets.MNIST(
            r'D:\Dataset\MNIST',
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )  # create dataloader
    model = Model().to(device)
    model.load_state_dict(torch.load("test_ckpt.pth", weights_only=True))
    test(model, device, test_loader)


if __name__ == '__main__':
    main()
