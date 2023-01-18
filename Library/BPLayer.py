
import numpy as np


# A Linear structure
class Linear:
    """
    Linear structure:
        input: X
        output: xw+w0
    demo:
        linear = Linear(inn=4, out=2)
        result = linear(X) # X.shape==(*,4)
    """
    def __init__(self, inn:int, out:int):
        self.inn_channels = inn 
        self.out_channels = out
        self.W = np.random.rand(inn+1,out)/4-0.125
        # Create W
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        bias = np.ones(X.shape[0]).reshape(-1,1)
        self.X = np.concatenate((bias, X),axis=1)
        # X={[1,x1,x2,...,xn],[1,x1,x2,...,xn],...}
        try: # This step is very easy to cause an error, but cannot find it.
            self.result = self.X @ self.W # y=X@w
        except ValueError: 
            raise ValueError("Ensure that the input data matches the model.")
        return self.result
    def backward(self, dldy, Lr):
        self.dydx = self.W.T
        # dy(output)/dx(input)=W.T
        self.dldx = (dldy @ self.dydx)
        # dl(loss)/dx(input)=[dl(loss)/dy(output)]@[dy(output)/dx(input)]
        self.update(dldy, Lr)
        # To updata W
        return self.dldx[:,1:]
    def update(self, dldy, Lr):
        self.dydw = self.X.T
        # dy(output)/dw(param)=X.T
        dldw = self.dydw @ dldy
        # dl(loss)/dw(param)=[dy(output)/dw(param)]@[dl(loss)/dy(output)]
        self.W = self.W - Lr*dldw
        # To updata W
    def get_grad_W(self):
        self.dydw = self.X.T # dy(output)/dw(param)=X.T
        return self.dydw
    def get_grad(self):
        self.dydx = self.W.T # dy(output)/dx(input)=W.T
        return self.dydx

# A Linear with regularization
class LinearRE(Linear):
    """
    Linear structure with regularization:
        input: X
        output: xw+w0
    demo:
        linear = Linear(inn=4, out=2, RElevel=0.001)
        result = linear(X) # X.shape==(*,4)
    """
    def __init__(self, inn:int, out:int, RElevel:float=0.0005):
        self.RElevel = RElevel
        super().__init__(inn=inn, out=out)
    def update(self, dldy, Lr):
        self.dydw = self.X.T
        # dy(output)/dw(param)=X.T
        dldw = self.dydw @ dldy + self.RElevel*self.W # +lambda*W (for regularization)
        # dl(loss)/dw(param)=[dy(output)/dw(param)]@[dl(loss)/dy(output)]
        self.W = self.W - Lr*dldw
        # To updata W
