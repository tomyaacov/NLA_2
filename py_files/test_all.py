import unittest
import numpy as np
from part_1_sd import steepest_decent
from part_1_cg import conjugate_gradient

MAX_ITER = 100
EPSILON = 0.0001


class MyTestCase(unittest.TestCase):

    def test_steepest_decent(self):
        A = np.array([
            [2, -1, 0],
            [-1, 2, -1],
            [0, -1, 2]
        ])
        b = np.array([0, -1, 6])
        x_0 = np.array([0, 0, 0])
        x, all_r = steepest_decent(A, b, x_0, MAX_ITER, EPSILON)
        real_x = np.linalg.solve(A, b)
        self.assertTrue(np.array_equal(np.around(x), np.around(real_x)))

    def test_conjugate_gradient(self):
        A = np.array([
            [2, -1, 0],
            [-1, 2, -1],
            [0, -1, 2]
        ])
        b = np.array([0, -1, 6])
        x_0 = np.array([0, 0, 0])
        x, all_r = conjugate_gradient(A, b, x_0, MAX_ITER, EPSILON)
        real_x = np.linalg.solve(A, b)
        self.assertTrue(np.array_equal(np.around(x), np.around(real_x)))


if __name__ == '__main__':
    unittest.main()
