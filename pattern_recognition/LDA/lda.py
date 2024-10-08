import numpy as np


class LDA:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.w = None

    def fit(self, X, y, tolerance=1e-6):
        total_dim = X.shape[-1]
        # get mu
        mean_vectors = []
        for label in np.unique(y):
            mean_vectors.append(np.mean(X[y == label], axis=0))
        mean_vectors = np.array(mean_vectors)
        # print(mean_vectors.shape)
        # get S_W
        S_W = np.zeros((total_dim, total_dim))
        for label, mean_vector in zip(np.unique(y), mean_vectors):
            class_scatter = X[y == label] - mean_vector
            S_W += class_scatter.T @ class_scatter
        # print(S_W.shape)
        # get S_B
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((total_dim, total_dim))
        for i, mean_vector in enumerate(mean_vectors):
            n = X[y == i].shape[0]
            class_scatter = mean_vector - overall_mean
            S_B += n * np.outer(class_scatter, class_scatter)
        # print(S_B.shape)
        # get eig
        S_W_inv = np.linalg.inv(S_W)
        eig_vals, eig_vecs = np.linalg.eig(S_W_inv @ S_B)
        assert np.all(np.abs(eig_vals.imag) < tolerance), "complex number in output of np.linalg.eig"
        eig_vals, eig_vecs = eig_vals.real, eig_vecs.real
        # print(eig_vals.shape, eig_vecs.shape)
        # get output
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        self.w = np.hstack([eig_pairs[i][1].reshape(X.shape[1], 1) for i in range(self.n_components)])


    def transform(self, X):
        return X.dot(self.w)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)