from numpy import asarray, array
from numpy.random import rand
from mnist import MNIST
import matplotlib.pyplot as plt
from py_files.part_5_b import gradient_test_reg_obj_grad_hes
from py_files.part_5_b import jacobian_test_reg_obj_grad_hes

# loading data
mndata = MNIST('py_files/data')
mndata.gz = True
images, labels = mndata.load_training()
images = asarray(images)
labels = asarray(labels)
X = images[labels<=1][:1000].T
X = X/X.max()
y = labels[labels<=1][:1000].T
w = rand(X.shape[0])*2-1

# running gradient test
res1, res2 = gradient_test_reg_obj_grad_hes(X, y, w, 1e-1, 12)

# plotting the decrease factor
plt.figure()
plt.plot([x[0]/x[1] for x in zip(array(res1)[:-1],array(res1)[1:])], label="Linear")
plt.plot([x[0]/x[1] for x in zip(array(res2)[:-1],array(res2)[1:])], label="Quadratic")
plt.legend()
plt.title("Gradient Test")
plt.ylabel("Decrease Factor")
plt.xlabel("Iteration")

plt.savefig("myplot4.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot4.pdf}")

# running jacobian test
res1, res2 = jacobian_test_reg_obj_grad_hes(X, y, w, 1e-1, 12)

# plotting the decrease factor
plt.figure()
plt.plot([x[0]/x[1] for x in zip(array(res1)[:-1],array(res1)[1:])], label="Linear")
plt.plot([x[0]/x[1] for x in zip(array(res2)[:-1],array(res2)[1:])], label="Quadratic")
plt.legend()
plt.title("Jacobian Test")
plt.ylabel("Decrease Factor")
plt.xlabel("Iteration")

plt.savefig("myplot5.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot5.pdf}")
