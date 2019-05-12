from numpy.random import rand
from numpy import matmul
from numpy.linalg import norm
from py_files.part_5_a import reg_obj_grad_hes


def gradient_test_reg_obj_grad_hes(X, y, w, epsilon, iterations):
    d = rand(w.shape[0])
    d = d/d.sum()
    res1 = []
    res2 = []
    obj_0, grad_0, _ = reg_obj_grad_hes(X, y, w)
    for _ in range(iterations):
        obj, grad, _ = reg_obj_grad_hes(X, y, w+(epsilon*d))
        res1.append(abs(obj-obj_0))
        res2.append(abs(obj-obj_0-epsilon*matmul(d.T,grad_0)))
        epsilon *= 0.5
    return res1, res2


def jacobian_test_reg_obj_grad_hes(X, y, w, epsilon, iterations):
    d = rand(w.shape[0])
    d = d/d.sum()
    res1 = []
    res2 = []
    obj_0, grad_0, hes_0 = reg_obj_grad_hes(X, y, w)
    for _ in range(iterations):
        obj, grad, _ = reg_obj_grad_hes(X, y, w+(epsilon*d))
        res1.append(norm(grad-grad_0))
        res2.append(norm(grad-grad_0-matmul(hes_0, epsilon*d.T)))
        epsilon *= 0.5
    return res1, res2

