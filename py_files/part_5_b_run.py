from numpy import asarray
from numpy.random import rand
from mnist import MNIST
import matplotlib.pyplot as plt
from py_files.part_5_b import gradient_test_reg_obj_grad_hes
from py_files.part_5_b import jacobian_test_reg_obj_grad_hes

try:
    mndata = MNIST('py_files/data')
    mndata.gz = True
    images, labels = mndata.load_training()
except FileNotFoundError:
    mndata = MNIST('data')
    mndata.gz = True
    images, labels = mndata.load_training()

images = asarray(images)
labels = asarray(labels)
X = images[labels <= 1][:100].T
X = X / X.max()
y = labels[labels <= 1][:100].T
w = rand(X.shape[0]) * 2 - 1

res1, res2 = gradient_test_reg_obj_grad_hes(X, y, w, 1e-4, 15)

plt.figure(figsize=(20, 25))
plt.plot(res1, label="Linear")
plt.plot(res2, label="Quadratic")
plt.legend()
plt.title("Gradient Test")
plt.xlabel("Iteration")

plt.savefig("myplot4.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot4.pdf}")

res1, res2 = jacobian_test_reg_obj_grad_hes(X, y, w, 1e-4, 15)

plt.figure(figsize=(20, 25))
plt.plot(res1, label="Linear")
plt.plot(res2, label="Quadratic")
plt.legend()
plt.title("Jacobian Test")
plt.xlabel("Iteration")

plt.savefig("myplot5.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot5.pdf}")
