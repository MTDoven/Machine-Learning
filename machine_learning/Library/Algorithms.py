
from Library.Criterion import *
from Library.Normalize import *
from Library.Activator import *
from Library.BPLayer import *
from Library.Module import *
import numpy as np


class LinearRegression(SequentialModule):
    """
    LinearRegression:
        This class can do linear regression automatically
        For more information, see the demo please.
    demo:
        linear_regression = LinearRegression(inn=4, out=1)
        loss = linear_regression.train(X, y, Lr=0.00007, epoch=1000, RElevel=0)
        y_pred = linear_regression.evaluate(X)
    """
    def __init__(self, inn, out):
        self.inn = inn; self.out = out
        self.Loss = MSELoss()
        self.Structure = [DataNormalize(), Linear(self.inn, self.out)]
    def __call__(self, X):
        return self.evaluate(X)
    def train(self, X, y, Lr=0.001, epoch=10000, RElevel=None):
        if RElevel is not None:
            self.Structure[1].RElevel = RElevel
        self.Lr = Lr
        self.target = y
        for _ in range(epoch):
            self.pred = self.forward(X)
            self.loss = self.Loss(self.pred,self.target)
            self.backward()
        return self.loss
    def evaluate(self, X):
        self.pred = self.forward(X)
        return self.pred


class LogisticRegression(SequentialModule):
    """
    LogisticRegression:
        This class can do logistic regression automatically
        For more information, see the demo please.
    demo:
        logistic_regression = LogisticRegression(inn=2, out=1)
        loss = logistic_regression.train(X, y, Lr=0.001, epoch=1000)
        y_pred = logistic_regression.evaluate(X)
    """
    def __init__(self, inn, out):
        self.inn = inn; self.out = out
        self.Loss = LogarithmicLoss()
        self.Structure = [DataNormalize(), LinearRE(self.inn,self.out), Sigmoid()]
    def __call__(self, X):
        return self.forward(X)
    def train(self, X, y, Lr=0.01, epoch=100):
        self.Lr = Lr
        self.target = y
        for _ in range(epoch):
            self.pred = self.forward(X)
            self.loss = self.Loss(self.pred,self.target)
            self.backward()
        return self.loss
    def evaluate(self, X):
        self.pred = self.forward(X)
        self.result = np.where(self.pred<0.5, 0, 1)
        return self.result


class PolynomialRegression(SequentialModule):
    """
    LogisticRegression:
        This class can do logistic regression automatically
        For more information, see the demo please.
    demo:
        polynomial_regression = PolynomialRegression(inn=2, out=1, power=6)
        loss = polynomial_regression.train(X, y, Lr=0.000001, epoch=50000, RElevel=0.001)
        y_pred = polynomial_regression.evaluate(X)
    """
    def __init__(self, inn, out, power):
        self.inn = inn
        self.out = out
        self.power = power
        self.Loss = MSELoss()
        self.Structure = [DataNormalize(), Polynomial(highest_term=power), LinearRE(power*self.inn,self.out)]
    def __call__(self, X):
        return self.evaluate(X)
    def train(self, X, y, Lr=0.001, epoch=10000, RElevel=0.0005):
        if RElevel is not None:
            self.Structure[1].RElevel = RElevel
        self.Lr = Lr
        self.target = y
        for _ in range(epoch):
            self.pred = self.forward(X)
            self.loss = self.Loss(self.pred,self.target)
            self.backward()
        return self.loss
    def evaluate(self, X):
        self.pred = self.forward(X)
        return self.result


class KMeans:
    """
    KMeans:
        This class can do KMeans automatically
        For more information, see the demo please.
    demo:
        kmeans = KMeans(K=3, max_iter=300)
        centers = kmeans.fit(X)
        classified_data = kmeans.get_classified_data()
    """
    def __init__(self, K:int, max_iter=300, relative_ol=0.0001):
        self.K = K
        self.max_iter = max_iter
        self.relative_ol = relative_ol
    def __call__(self, X):
        return self.fit(X)
    def get_classified_data(self):
        return self.classified_data
    def fit(self, X):
        assert len(X)>=self.K, "K is larger than the number of data"
        assert len(X.shape)==2, "Shape of X is not in 2D."
        self.X = X # save X for other use
        self.classified_data = []  # To save X which has been classified
        def cal_the_cluster(all_X, centers:list):
            distances = []
            for center in centers:
                temp = np.sum(np.square(all_X-center),axis=1)
                distances.append(temp)
            distances = np.array(distances)
            minimum = np.argmin(distances.T, axis=1)
            return minimum
        def update_the_center(all_X, minimum:np.ndarray):
            self.classified_data = []
            for i in range(self.K):
                index = np.argwhere(minimum!=i).reshape(-1)
                x = np.delete(all_X,index,axis=0)
                self.classified_data.append(x)
            new_centers = [np.mean(x,axis=0) for x in self.classified_data]
            return new_centers
        # start the algorithm
        self.centers = self.choose_center()
        for _ in range(self.max_iter):
            minimum_label = cal_the_cluster(self.X, self.centers)
            temp_centers = update_the_center(self.X, minimum_label)
            if np.allclose(np.array(temp_centers),np.array(self.centers), rtol=self.relative_ol):
                self.centers = temp_centers
                break # If it has been great enough, then exit
            self.centers = temp_centers
        return self.centers
    def choose_center(self)->list:
        def cal(all_X, centers:list):
            _cal_distance_to_all_X = lambda x2:np.sum(np.square(all_X-x2),axis=1)
            result = tuple(map(_cal_distance_to_all_X,centers))
            try: # To avoid error caused by len(result)==1
                minimum = np.minimum(result[0],result[1])
            except IndexError:
                return result[0]
            for i in range(2,len(result)):
                minimum = np.minimum(minimum,result[i])
            return minimum
        centers = [] # Create a list to save the centers
        temp = np.random.randint(len(self.X))
        for _ in range(self.K):
            temp = self.X[temp] #temp:int->ndarray
            centers.append(temp) #temp:ndarray: add to center
            temp = cal(self.X, centers) #temp:ndarray: is distance
            temp = np.argmax(temp) #temp:ndarray->int
        return centers


class SVM:
    """
    !!! There are still something wrong in it, but it can relatively work well !!!
    Support Vector Machine:
        This class can do SVM automatically
        For more information, see the demo please.
    demo:
        svm = SVM(C=100, toler=0.001, iter_max=100)
        svm.fit(X, y)
        W = svm.get_W()  # [b,w1,w2,...]
        result = svm.evaluate(X)
    attention: calculate_kernel_results(self):
        This Kernel function could be rewritten.
        from self.dataset return (self.N x self.N) matrix (sample length by sample length !not! dim)
        :return: <ndarray.float64>
    """
    def __init__(self, C, toler, iter_max):
        self.C = C
        self.toler = toler
        self.iter_max = iter_max
    def calculate_kernel_results(self):
        """
        This Kernel function could be rewritten.
        from self.dataset return (self.N x self.N) matrix (sample length by sample length !not! dim)
        :return: <ndarray.float64>
        """
        return np.matmul(self.dataset,self.dataset.T)
    def get_W(self):
        return self.W
    def __call__(self,X):
        return self.evaluate(X)
    def evaluate(self, X):
        self.result = self.linear(X)
        self.result = np.where(self.result>0, 1, 0)
        #self.result = np.where(self.result>0, 1, 0)
        return self.result
    def prepare_for_evaluate(self):
        self.linear = Linear(self.M, 1)
        self.W = np.array([self.b] + list(self.w)).reshape(-1, 1)
        self.linear.W = self.W
    def fit(self, dataset, target):
    ############################# initiate some dataset and params ###################################
        self.dataset = dataset # save dataset
        self.y = np.array([1 if i == 1 else -1 for i in target.reshape(-1)])  # Transform y(0,1) to y(-1,1)
        self.Kernel_result = self.calculate_kernel_results() # calculate Kernel matrix: K[ij]
        self.N, self.M = dataset.shape # N:samples; M:dims
        self.Alpha = np.zeros(self.N) # initiate alpha as 0
        self.b = 0 # initiate b as 0
    ###################################### some useful function ######################################
        get_Ex = lambda i: np.sum(self.Alpha * self.y * self.Kernel_result[i]) + self.b - self.y[i]
        # calculate Ei or Ej, (writen as Ex)
        kernel = lambda i,j: self.Kernel_result[i, j]
        # return the kernel from self.Kernel_result
        def to_check_and_clip_alpha(L, H, alpha_j):
            # Clip alpha, if alpha isn't fit the request
            if alpha_j < L: alpha_j = L
            elif alpha_j > H: alpha_j = H
            return alpha_j
        def meet_KKT_conditions(ei, yi, alphai):
            # if meet KKT conditions return True otherwise return False
            if yi*ei < -self.toler and alphai < self.C:
                return False
            if yi*ei > self.toler and alphai > 0:
                return False
            return True
        def get_a_random_i(): # generator
            # get i from a sequential method, (this function can be updated)
            number_listt = [i for i in range(self.N)]
            while True: # always go around
                for i in number_listt: yield i
        def get_j_from_fixed_i(): # generator
            # get j from a random method, (this function can be updated)
            nonlocal i # i is the outside one
            number_listt = [i for i in range(self.N)]
            while True: # always go around
                random.shuffle(number_listt)
                for j in number_listt:
                    if j!=i: yield j
        def get_Low_and_High(i, j):
            # get low edge and high edge of alpha
            if self.y[i] != self.y[j]:
                L = max([0, self.Alpha[j]-self.Alpha[i]])
                H = min([self.C, self.C+self.Alpha[j]-self.Alpha[i]])
            else:  # self.y[i] == self.y[j]
                L = max([0, self.Alpha[j]+self.Alpha[i]-self.C])
                H = min([self.C, self.Alpha[i]+self.Alpha[j]])
            return L, H
    ####################################### main circle #############################################
        iter_recorder = 0
        get_next_i = get_a_random_i() # create generator for i
        get_next_j = get_j_from_fixed_i() # create generator for j
        while iter_recorder < self.iter_max:
            have_updated_params = False # stop condition used
            for _ in range(self.N):
    ########################### initiate first value i and second value j ###########################
                i = next(get_next_i)
                Ei = get_Ex(i) # get a Ei as the first Ei
                if meet_KKT_conditions(Ei, self.y[i], self.Alpha[i]):
                    continue # choose those oppose KKT conditions to calculate and to update
                j = next(get_next_j)
                Ej = get_Ex(j)
    ########################## calculate proper range of alpha_j ####################################
                alpha_i = self.Alpha[i] # alpha_i_old
                alpha_j = self.Alpha[j] # alpha_j_old
                # record alpha i and j
                Low, High = get_Low_and_High(i, j)
                if Low > High: # wrong situation
                    continue # maybe a bug caused by random j ?
                eta = kernel(i,i) + kernel(j,j) - 2*kernel(i,j)
                if eta <= 0: # wrong situation
                    continue # maybe a bug caused by random j ?
    #################################### updata alpha_j #############################################
                self.Alpha[j] += self.y[j] * (Ei-Ej)/eta
                self.Alpha[j] = to_check_and_clip_alpha(Low, High, self.Alpha[j]) # updata alpha_j
                tempj = self.Alpha[j] - alpha_j
                tempi = self.Alpha[i] - alpha_i
                if abs(tempj)<self.toler**2:
                    continue # alpha is good enough
                self.Alpha[i] += self.y[i] * self.y[j] * (-tempj) # update Alpha_i
    ######################################## update b ###############################################
                if 0 < self.Alpha[i] < self.C:
                    self.b = self.b - Ei - self.y[i]*kernel(i,i)*tempi - self.y[j]*kernel(i,j)*tempj
                elif 0 < self.Alpha[j] < self.C:
                    self.b = self.b - Ej - self.y[i]*kernel(i,j)*tempi - self.y[j]*kernel(j,j)*tempj
                else: # (b1 == 0 or C) and (b2 == 0 or C)
                    b1 = self.b - Ei - self.y[i]*kernel(i,i)*tempi - self.y[j]*kernel(i,j)*tempj
                    b2 = self.b - Ej - self.y[i]*kernel(i,j)*tempi - self.y[j]*kernel(j,j)*tempj
                    self.b = (b1+b2)/2.0
                have_updated_params = True
    ################################### stop condition used #########################################
            iter_recorder = iter_recorder+1 if not have_updated_params else 0
    ################################## calculate W from alpha #######################################
        self.w = (self.Alpha * self.y) @ self.dataset # Calculate W
        self.prepare_for_evaluate() # Calculate W(with b in it) and create a linear model

