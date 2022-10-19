import numpy as np
from robustopt.dro import drlr_cxvpy
from robustopt.dro import lr

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def label(y):
    l = []
    for i in y:
        if i == 0:
            l.append(-1)
        if i == 1:
            l.append(1)
    return np.array(l)

def train_dataset():
    M = 50
    N = 100
    beta_true = np.array([1, 0.5, -0.5] + [0]*(M - 3))
    x = (np.random.random((N, M)) - 0.5)*10
    beta_true = beta_true.reshape(M,1)
    x = x.reshape(N,M)

    y = label(np.round(sigmoid(x@beta_true)))
    y = y.reshape(N,1)

    for i in np.random.randint(1,N,(20,1)):
        y[i] = -1
    for i in np.random.randint(1,N,(20,1)):
        y[i] = 1
    return x,y,beta_true

def test_dataset():
    M = 50
    N = 100
    beta_true = np.array([1, 0.5, -0.5] + [0]*(M - 3))
    x = (np.random.random((N, M)) - 0.5)*10
    beta_true = beta_true.reshape(M,1)
    x = x.reshape(N,M)

    y = label(np.round(sigmoid(x@beta_true)))
    y = y.reshape(N,1)
    return x,y,beta_true

def main():
    x_train,y_train,_ = train_dataset()
    x_test ,y_test ,_ = test_dataset()

    e = 0
    k = 0
    pn= 1 
    pv= np.inf
    for epsilon in [1e-3,1e-1,1,1e1,1e3]:
        for kappa in [1e-3,1e-1,1,1e1,1e3]:
            for pnorm in [1,2]:
                try:
                    drlr = drlr_cxvpy.dr_logistic_regression(epsilon,kappa,pnorm)
                    p_value,_,_,_ = drlr.fit(x_train,y_train)
                    if p_value < pv:
                        pv = p_value
                        e = epsilon
                        k = kappa
                        pn= pnorm
                except:
                    pass
    drlr = drlr_cxvpy.dr_logistic_regression(epsilon=e,kappa=k,pnorm=pn)
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
