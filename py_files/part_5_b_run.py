from numpy import asarray, array
from numpy.random import rand
from mnist import MNIST
import matplotlib.pyplot as plt
from py_files.part_5_b import gradient_test_reg_obj_grad_hes
from py_files.part_5_b import jacobian_test_reg_obj_grad_hes

X = array([[1, 0, 1], [0, 1, 1]]).T
y = array([0, 1])
w = array([1, 1, 0])

res1, res2 = gradient_test_reg_obj_grad_hes(X, y, w, 1e-1, 15)

plt.figure()
plt.plot(res1, label="Linear")
plt.plot(res2, label="Quadratic")
plt.legend()
plt.title("Gradient Test")
plt.xlabel("Iteration")

plt.savefig("myplot4.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot4.pdf}")

res1, res2 = jacobian_test_reg_obj_grad_hes(X, y, w, 1e-4, 15)

plt.figure()
plt.plot(res1, label="Linear")
plt.plot(res2, label="Quadratic")
plt.legend()
plt.title("Jacobian Test")
plt.xlabel("Iteration")

plt.savefig("myplot5.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot5.pdf}")
