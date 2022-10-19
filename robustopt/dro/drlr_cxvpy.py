import numpy as np
import cvxpy as cp
from sklearn.metrics import confusion_matrix

class dr_logistic_regression():
  def __init__(self,epsilon,kappa,pnorm):
    self.epsilon = epsilon
    self.kappa   = kappa
    self.pnorm   = pnorm

  def fit(self,x,y):
    self.N = x.shape[0]
    self.M = x.shape[1]
    self.x = x
    self.y = y
    beta_   = cp.Variable((self.M,1))
    lambda_ = cp.Variable((1,1))
    s       = cp.Variable((self.N,1))

    constraints = [
                 cp.logistic( cp.multiply(-self.y,self.x@beta_) ) <= s 
                ,cp.logistic( cp.multiply(+self.y,self.x@beta_) ) - lambda_*self.kappa <= s
                ,cp.norm(beta_, self.pnorm)<=lambda_
                  ]

    objective = lambda_*self.epsilon + 1/self.N*cp.sum(s)

    problem = cp.Problem(cp.Minimize(objective),constraints)
    problem.solve(verbose=False)
    self.beta_ = beta_.value
    return problem.value,beta_.value,lambda_.value,s.value
    
  def infer(self,x):
      y_est = []
      for i in 1/(1+np.exp(-x@self.beta_)):
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
