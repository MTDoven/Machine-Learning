
import numpy as np
import random


class DataNormalize:
    """
    DataNormalize:
        input: X
        output: (X-mean(X))/std(X)
    demo:
        dn = DataNormalize()
        X_normed = dn.forward(X)
        X_restored = dn.restore(X_normed)
        dydx = dn.get_grad() # d(X_normed)/d(X)
    """
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.output = (X-self.mean)/self.std
        # y=(X-mean)/std
        return self.output
    def backward(self, dldy, Lr):
        self.dydx = 1/self.std
        # dy/dx=1/std
        return self.dydx * dldy
    def restore(self, output):
        self.input = output*self.std + self.mean
        # X=y*std+mean
        return self.input
    def get_grad(self):
        self.dydx = 1/self.std
        return self.dydx

    
class KernelTransform:
    """
    DataNormalize:
        input: [x1, x2, ...]
        output: [f1, f2, ...]
    demo:
        kt = KernelTransform(kernel_number=len(X)*2)
        X_transformed = kt.forward(X)
    """
    def __init__(self, kernel_number=None, sigma=None):
        self.kernel_number = kernel_number
        self.sigma = sigma
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        self.X_origin = X.copy()
        X = DataNormalize()(X)
        if self.kernel_number is None: 
            self.kernel_number = len(X)//5
        if self.sigma is None:
            self.sigma = np.mean(np.std(X, axis=0))
        self.kernels = np.array(random.sample(list(X), self.kernel_number))
        self.similarity = lambda x,y: np.exp(-np.sum(np.square(x-y),axis=1)/(2*self.sigma**2))
        self.output = np.array([self.similarity(i,self.kernels) for i in X])
        return self.output


class PCA:
    """
    PCA:
        input: [x1, x2, ..., xn]
        output: [x1, x2, ..., xr] # r<n
    demo:
        pca = PCA(inn=X.shape[-1], out=3)
        result = pca.forward(X)
        X_restored = pca.restore(result)
    """
    def __init__(self, inn, out):
        self.n = inn
        self.k = out
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        self.X_origin = X.copy()
        X = DataNormalize()(X)
        Sigma = (X.T @ X)/len(X)
        self.U,_,_ = np.linalg.svd(Sigma)
        try: # avoid k>n
            self.U_reduce = self.U[:,:self.k]
        except IndexError:
            raise AssertionError("K is larger than dim:n.")
        self.result = X @ self.U_reduce
        return self.result
    def restore(self, y):
        return y @ self.U_reduce.T

