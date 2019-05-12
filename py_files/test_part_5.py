from numpy import matmul, exp, log, asarray, outer, array
from numpy.random import rand
from mnist import MNIST
from py_files.part_5_a import reg_obj_grad_hes

mndata = MNIST('py_files/data')
mndata.gz = True
images, labels = mndata.load_training()
images = asarray(images)
labels = asarray(labels)
X = images[labels<=1][:4].T
X = X/X.max()
y = labels[labels<=1][:4].T
w = rand(X.shape[0])*2-1
print(reg_obj_grad_hes(X, y, w))


import numpy as np

X = np.array([[1, 0, 1], [0, 1, 1]])
y = np.array([0, 1])
w = np.array([1, 1, 0])
print(reg_obj_grad_hes(X.T, y, w))

def gradient_test(iterations=10):
    X = np.array([[1, 0, 1], [0, 1, 1]])
    labels = np.array([0, 1])
    w = np.array([1, 1, 0], dtype=np.float64)
    d = np.random.rand(w.shape[0])
    d = d/d.sum()

    res1 = []
    res2 = []

    classifier_x = LogisticRegression(samples=X, labels=labels, w=w)
    loss_0 = classifier_x.loss()
    grad_0 = classifier_x.gradient_logistic_regression()

    for _ in range(iterations):
        epsilon = 1
        w_i = w + (epsilon * d)
        classifier = LogisticRegression(samples=X, labels=labels, w=w_i)
        loss = classifier.loss()
        res1.append(abs(loss - loss_0))
        res2.append(abs(loss - loss_0 - epsilon*np.dot(d.T,grad_0)))
        epsilon *= 0.5

    return res1, res2
