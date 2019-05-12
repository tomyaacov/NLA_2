from numpy import matmul, array, dot, copy, asarray, mean, round, eye
from numpy.linalg import norm, inv
from numpy.random import rand
from mnist import MNIST
import matplotlib.pyplot as plt
from py_files.utils import is_pos_def
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
        w = w - matmul(inv(hes+eye(hes.shape[0])*epsilon), grad)
        f.append(obj)
        if len(f) > 1 and f[-2] - obj < epsilon:
            break
    return w, f


def accuracy(X, y, w):
    return mean(y == round(sigmoid(matmul(X.T, w))))
