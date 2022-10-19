import numpy as np
import cvxpy as cp
from sklearn.metrics import confusion_matrix

class logistic_regression():
    def __init__(self,pnorm,lambd):
        self.pnorm = pnorm
        self.lambd = lambd

    def fit(self,x,y):
        self.N = x.shape[0]
        self.M = x.shape[1]
        self.x = x
        self.y = y
        beta = cp.Variable((self.M,1))
        
        log_likelihood = cp.sum(cp.logistic( cp.multiply(-self.y,self.x@beta) ))
        problem = cp.Problem(cp.Minimize(log_likelihood/self.N + self.lambd * cp.norm(beta, self.pnorm)))
        problem.solve(verbose=False)
        self.beta = beta.value
        return problem.value,beta.value

    def infer(self,x):
        y_est = []
        for i in 1/(1+np.exp(-x@self.beta)):
            if i>=0.5:
                y_est.append(1)
            if i<0.5:
                y_est.append(-1)
        self.y_est = y_est
        return y_est
    
    @staticmethod
    def confusion_matrix(y,y_est):
        C = confusion_matrix(y, y_est)
        return C