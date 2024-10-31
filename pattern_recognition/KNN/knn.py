import torch
from torchvision import datasets, transforms
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    @torch.no_grad()
    def fit(self, X, y, device="cuda"):
        self.X_train = X.to(device)
        self.y_train = y.to(device)

    @torch.no_grad()
    def _compute_distance(self, x):
        if self.distance_metric == 'euclidean':
            return torch.cdist(x.unsqueeze(0), self.X_train, p=2)
        elif self.distance_metric == 'manhattan':
            return torch.cdist(x.unsqueeze(0), self.X_train, p=1)
        elif self.distance_metric == 'cosine':
            x_norm = F.normalize(x.unsqueeze(0))
            X_train_norm = F.normalize(self.X_train, dim=1)
            return 1 - torch.mm(x_norm, X_train_norm.T)
        else:  # NotImplementedError
            raise NotImplementedError(f"Unknown distance metric: {self.distance_metric}")

    @torch.no_grad()
    def predict(self, X):
        X = X.to(self.X_train.device)
        predictions = []
        for x in X:
            distances = self._compute_distance(x)
            k_indices = torch.topk(distances, self.k, largest=False).indices
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels.view(-1).tolist()).most_common(1)
            predictions.append(most_common[0][0])
        return torch.tensor(predictions)


def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST(root=r"D:\Dataset\MNIST", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=r"D:\Dataset\MNIST", train=False, download=True, transform=transform)
    return train_set, test_set


def load_cifar():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root=r"D:\Dataset\CIFAR10", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=r"D:\Dataset\CIFAR10", train=False, download=True, transform=transform)
    return train_set, test_set


def load_iris():
    iris = sk_datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
