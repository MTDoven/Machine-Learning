
from Library.Normalize import DataNormalize


class SequentialModule:
    """
    SequentialModule:
        input: X
        output: result
    demo:
        class Model(SequentialModule):
            def __init__(self, loss):
                self.Structure = [Linear(4,4), Linear(4,1), Sigmoid()]
                self.Lr = 0.00001
                self.Loss = loss
        # Ensure there are self.Structure, self.Lr, self.Loss in your model
        criterian = LogarithmicLoss()
        model = Model(criterion)
        # criterion function should be input in it, or you can initiate in the class Model
        for _ in range(10000):
            result = model(X)
            loss = criterion(result,y)
            model.backward()
    attention:
        Ensure there are self.Structure, self.Lr, self.Loss in your model
    """
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        X = DataNormalize()(X)
        for i in self.Structure:
            X = i.forward(X)
        self.result = X
        return self.result
    def backward(self):
        dldz = self.Loss.get_grad()
        for i in reversed(self.Structure):
            dldz = i.backward(dldz, self.Lr)

