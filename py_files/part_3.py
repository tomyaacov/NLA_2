from numpy import matmul, array, dot, copy, transpose, vectorize
from numpy.linalg import norm
from utils import is_pos_def
import matplotlib.pyplot as plt


def steepest_decent(A, b, x_0, max_iter, epsilon):
    if not is_pos_def(A):
        print("matrix is not SPD, can't solve using steepest decent...")
        return None, None
    x = copy(x_0)
    r = b - matmul(A, x)
    all_r = [norm(r)]
    for k in range(max_iter):
        alph = dot(r, matmul(A, r)) / matmul(r, matmul(transpose(A), matmul(A, r)))
        x = x + alph * r
        r = b - matmul(A, x)
        all_r.append(norm(r))
        if norm(r) / norm(b) < epsilon:
            break
    return x, array(all_r)


if __name__ == '__main__':
    A = array([
        [5, 4, 4, -1, 0],
        [3, 12, 4, -5, -5],
        [-4, 2, 6, 0, 3],
        [4, 5, -7, 10, 2],
        [1, 2, 5, 3, 10]
    ])
    b = array([1, 1, 1, 1, 1])
    x_0 = array([0, 0, 0, 0, 0])
    x, all_r = steepest_decent(A, b, x_0, 50, 0.00000000001)
    plt.figure(figsize=(20, 20))
    plt.semilogy(all_r)
    plt.savefig("part_3.png")


