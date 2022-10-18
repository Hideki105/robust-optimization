import numpy as np
from robustopt import drlr_cxvpy
from robustopt import lr

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def train_dataset():
    M = 100
    N = 500
    beta_true = np.array([1, 0.5, -0.5] + [0]*(M - 3))
    x = (np.random.random((N, M)) - 0.5)*10
    beta_true = beta_true.reshape(M,1)
    x = x.reshape(N,M)

    y = np.round(sigmoid(x@beta_true))
    y = y.reshape(N,1)
    return x,y,beta_true

def test_dataset():
    M = 100
    N = 200
    beta_true = np.array([1, 0.5, -0.5] + [0]*(M - 3))
    x = (np.random.random((N, M)) - 0.5)*10
    beta_true = beta_true.reshape(M,1)
    x = x.reshape(N,M)

    y = np.round(sigmoid(x@beta_true))
    y = y.reshape(N,1)
    return x,y,beta_true

def main():
    x_train,y_train,beta_true = train_dataset()
    x_test,y_test,beta_true = test_dataset()

    epsilon = 1e-3
    kappa = 1
    pnorm = 1
    drlr = drlr_cxvpy.dr_logistic_regression(epsilon,kappa,pnorm)
    drlr.fit(x_train,y_train)
    
    y_est = drlr.infer(x_train)
    C = drlr.confusion_matrix(y_train,y_est)
    print(C)
    
    y_est = drlr.infer(x_test)
    C = drlr.confusion_matrix(y_test,y_est)
    print(C)

    pnorm = 2
    lambd = 1e-3
    lr_ = lr.logistic_regression(pnorm,lambd)
    lr_.fit(x_train,y_train)
    
    y_est = lr_.infer(x_train)
    C = lr_.confusion_matrix(y_train,y_est)
    print(C)
    
    y_est = lr_.infer(x_test)
    C = lr_.confusion_matrix(y_test,y_est)
    print(C)

if __name__ == "__main__":
    main()
