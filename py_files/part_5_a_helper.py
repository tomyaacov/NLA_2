from numpy import matmul, exp, log


def sigmoid(x):
    return 1 / (1 + exp(-x))


def reg_obj(X, y, w):
    m = len(y)
    c_1 = y
    c_2 = 1-y
    sig = sigmoid(matmul(X.T,w))
    obj = -(1/m)*(matmul(c_1.T,log(sig))+matmul(c_2.T,log(1-sig)))
    return obj


