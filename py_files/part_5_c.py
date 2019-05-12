from numpy import matmul, copy, mean, round, eye
from numpy.linalg import inv
from py_files.part_5_a import reg_obj_grad_hes, sigmoid


def steepest_decent(X, y, w_0, max_iter, epsilon, alph):
    w = copy(w_0)
    f = []
    for _ in range(max_iter):
        obj, grad, _ = reg_obj_grad_hes(X, y, w)
        w = w - alph * grad
        f.append(obj)
        if len(f) > 1 and f[-2] - obj < epsilon:
            break
    return w, f


def newton(X, y, w_0, max_iter, epsilon):
    w = copy(w_0)
    f = []
    for _ in range(max_iter):
        obj, grad, hes = reg_obj_grad_hes(X, y, w)
        w = w - matmul(inv(hes+eye(hes.shape[0])), grad)
        f.append(obj)
        if len(f) > 1 and f[-2] - obj < epsilon:
            break
    return w, f


def accuracy(X, y, w):
    return mean(y == round(sigmoid(matmul(X.T, w))))
