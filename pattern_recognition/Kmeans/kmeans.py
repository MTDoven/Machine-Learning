import torch


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, device='cuda'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device if torch.cuda.is_available() else 'cpu'

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        num_samples, num_features = X.shape
        indices = torch.randperm(num_samples)[:self.n_clusters]
        centers = X[indices]
        for _ in range(self.max_iter):
            distances = torch.cdist(X, centers)
            labels = torch.argmin(distances, dim=1)
            new_centers = torch.stack([X[labels == i].mean(dim=0) for i in range(self.n_clusters)])
            shift = torch.norm(new_centers - centers)
            if shift < self.tol:
                break
            centers = new_centers
        self.centers = centers
        self.labels = labels.cpu().numpy()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        distances = torch.cdist(X, self.centers)
        labels = torch.argmin(distances, dim=1)
        return labels.cpu().numpy()

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
