from numpy import matmul, exp, log, asarray, outer
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
