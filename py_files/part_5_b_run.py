from numpy import array
import matplotlib.pyplot as plt
from py_files.part_5_b import gradient_test_reg_obj_grad_hes
from py_files.part_5_b import jacobian_test_reg_obj_grad_hes

# creating test example
X = array([[1, 0],
        [0, 1],
        [1, 1]])
y = array([0, 1])
w = array([1, 1, 0])

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
