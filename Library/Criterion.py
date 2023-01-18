
import numpy as np


class MSELoss:
    """
    MSELoss:
        input: y, target
        output: loss, d(loss)/d(y)
    demo:
        criterion = MSELoss()
        loss = criterion(y, target)
        dldy = criterion.get_grad()
    """
    def __call__(self, result, target):
        self.error = result-target
        # error = result-y
        self.loss = np.mean(np.square(self.error))/2
        # (1/2n)*sum^(1~n){(result-y)^2}
        self.dldz = self.error #/ len(result)
        # (result-y)/n
        return self.loss
    def get_grad(self):
        return self.dldz

class LogarithmicLoss:
    """
    BCELoss:
        input: y, target
        output: loss, d(loss)/d(y)
    demo:
        criterion = BCELoss()
        loss = criterion(y, target)
        dldy = criterion.get_grad()
    """
    def __init__(self):
        pass
    def __call__(self, result, target):
        self.loss = - np.mean( target*np.log(result) + (1-target)*np.log(1-result) )
        # -(1/n)*sum^(1~n){(y)*ln(result)+(1-y)*ln(1-result)}
        self.dldz = - target*(1/result) + (1-target)*(1/(1-result))
        # -(1/n)*{-(y)*[1/result]+(1-y)*[1/(1-result)]}
        return self.loss
    def get_grad(self):
        return self.dldz