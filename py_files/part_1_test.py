import numpy as np
from py_files.part_1_gauss_seidel import weighted_gauss_seidel
from py_files.part_1_jacobi import weighted_jacobi

A = np.array([[4, -1, 1], [4, -8, 1], [-2, 1, 5]])
b = np.array([7, -21, 15])
x_0 = np.array([1, 2, 2])
print(weighted_jacobi(A, b, x_0, 100, 1e-5, 1))
print(weighted_gauss_seidel(A, b, x_0, 100, 1e-5, 1))
