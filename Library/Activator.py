
import numpy as np


class Sigmoid:
    """
    Sigmoid function:
        input: x
        output: 1/[1+exp(-x)]
    demo:
        sigmoid = Sigmoid()
        result = sigmoid(X) # X is input data
    """
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        self.output = 1/(1+np.exp(-X))
        # y = Sigmoid(x) = 1/[1+exp(-x)]
        return self.output
    def backward(self, dldy, Lr):
        self.dydx = self.output * (1 - self.output)
        # grad(x) = y*(1-y)
        return self.dydx * dldy
    def get_grad(self):
        self.dydx = self.output * (1 - self.output)
        return self.dydx



class Polynomial:
    """
    Polynomial function:
        input: [x1, x2]
        output: [x1, x2, x1^2, x2^2, x1^3, x2^3 ...]
    demo1:
        pol = Polynomial(highest_term=3)
        result = pol(X)
    demo2:
        pol = Polynomial(functions=(function1,function2,...), gradfuncs:(gradfuncs1,gradfuncs2,...))
        result = pol(X)
    """
    def __init__(self, highest_term:int=None, functions:tuple=None, gradfuncs:tuple=None):
        assert (functions is None)==(gradfuncs is None), "grad function is different from forward function"
        assert (highest_term is not None) or (functions is not None), "Set the output please."
        def create_functions(i): return lambda x:x**i
        def create_gradfuncs(i): return lambda x:i*x**(i-1)
        if functions is None:
            self.functions = [create_functions(i) for i in range(1, highest_term+1)]
            self.gradfuncs = [create_gradfuncs(i) for i in range(1, highest_term+1)]
        else: # function is None
            self.functions = functions
            self.gradfuncs = gradfuncs
        self.highest_term = highest_term
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        self.X = X # save X
        temp = [i(X) for i in self.functions]
        self.result = np.concatenate(temp, axis=1)
        # [X] to [X^1,X^2,X^3,...,X^n]
        return self.result
    def backward(self, dldy, Lr):
        self.ys = tuple(i(self.X) for i in self.gradfuncs)
        self.dydx = np.concatenate(self.ys, axis=1)
        # All dy/dx=[1, 1, 2*X1^1, 2*X2^1, 3*X1^2, 3*X2^2, 4*X1^3, 4*X2^3]
        self.dldx = dldy * self.dydx
        # All dl/dx=[dl/dx1, dl/dx2, dl/dx1, dl/dx2, dl/dx1, dl/dx2, dl/dx1, dl/dx2]
        self.dldx = sum(np.split(self.dldx, self.highest_term, axis=1))
        # dl/dx=[sum(dl/dx1), sum(dl/dx2)]
        return self.dldx
    def get_grad(self):
        self.ys = tuple(i(self.X) for i in self.gradfuncs)
        self.dydx = np.concatenate(self.ys, axis=1)
        # All dy/dx=[1, 1, 2*X1^1, 2*X2^1, 3*X1^2, 3*X2^2, 4*X1^3, 4*X2^3]
        return self.dydx