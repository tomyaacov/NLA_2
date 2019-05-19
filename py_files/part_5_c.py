from numpy import matmul, copy, mean, round, eye
from numpy.linalg import inv
from py_files.part_5_a import reg_obj_grad_hes, sigmoid
from py_files.part_5_a_helper import reg_obj


def steepest_decent(X, y, w_0, max_iter, alph_0, beta=0.5, c=1e-4):
    w = copy(w_0)
    w_l = [w]
    for _ in range(max_iter):
        obj, grad, _ = reg_obj_grad_hes(X, y, w)
        alph = armijo_line_search(X, y, w, obj, grad, -grad, alph_0, beta, c)
        if alph is None:
            break
        w = w - alph * grad
        w_l.append(w)
    return w_l


def newton(X, y, w_0, max_iter, epsilon):
    w = copy(w_0)
    f = []
    w_l = [w]
    for _ in range(max_iter):
        obj, grad, hes = reg_obj_grad_hes(X, y, w)
        w = w - matmul(inv(hes+eye(hes.shape[0])), grad)
        w_l.append(w)
        f.append(obj)
        if len(f) > 1 and f[-2] - obj < epsilon:
            break
    return w_l


def armijo_line_search(X, y, w, obj, grad, d, alpha, beta=0.5, c=1e-4, max_iter=100):
    n_obj = reg_obj(X, y, w + alpha * d)
    for _ in range(max_iter):
        if n_obj <= obj + c * alpha * matmul(grad, d):
            return alpha
        else:
            alpha = beta * alpha
    return None


def accuracy(X, y, w):
    return mean(y == round(sigmoid(matmul(X.T, w))))
