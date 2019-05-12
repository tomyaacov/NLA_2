from numpy import asarray
from numpy.random import rand
from mnist import MNIST
import matplotlib.pyplot as plt
from py_files.part_5_c import steepest_decent, newton, accuracy

# loading data


try:
    mndata = MNIST('py_files/data')
    mndata.gz = True
    images_train, labels_train = mndata.load_training()
    images_test, labels_test = mndata.load_testing()
except FileNotFoundError:
    mndata = MNIST('data')
    mndata.gz = True
    images_train, labels_train = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

# parameters definition
max_iter = 1000
epsilon = 1e-4
alph = 0.05

# 0/1

# train data pre processing
images_train = asarray(images_train)
labels_train = asarray(labels_train)
X_train = images_train[labels_train <= 1].T
X_train = X_train / X_train.max()
y_train = labels_train[labels_train <= 1].T

# test data pre processing
images_test = asarray(images_test)
labels_test = asarray(labels_test)
X_test = images_test[labels_test <= 1].T
X_test = X_test / X_test.max()
y_test = labels_test[labels_test <= 1].T

# initializing weights
w_0 = rand(X_train.shape[0]) * 2 - 1

# running sd an newton
w_sd, f_sd = steepest_decent(X_train, y_train, w_0, max_iter, epsilon, alph)
w_n, f_n = newton(X_train, y_train, w_0, max_iter, epsilon)

# computing sd accuracy
print(r"Steepest Decent Accuracy:\\")
print(r"Train:", accuracy(X_train, y_train, w_sd), r"\\")
print(r"Test:", accuracy(X_test, y_test, w_sd), r"\\")

# computing newton accuracy
print(r"Newton Accuracy:\\")
print(r"Train:", accuracy(X_train, y_train, w_n), r"\\")
print(r"Test:", accuracy(X_test, y_test, w_n), r"\\")

# plotting convergence history
plt.figure()
plt.plot(f_sd, label="Steepest Decent")
plt.plot(f_n, label="Newton")
plt.legend()
plt.title("0/1 Logistic Regression Convergence History")
plt.ylabel("Logistic Regression Objective")
plt.xlabel("Iteration")

plt.savefig("myplot6.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot6.pdf}")

# 8/9

# train data pre processing
images_train = asarray(images_train)
labels_train = asarray(labels_train)
X_train = images_train[labels_train >= 8].T
X_train = X_train / X_train.max()
y_train = labels_train[labels_train >= 8].T
y_train -= 8

# test data pre processing
images_test = asarray(images_test)
labels_test = asarray(labels_test)
X_test = images_test[labels_test >= 8].T
X_test = X_test / X_test.max()
y_test = labels_test[labels_test >= 8].T
y_test -= 8

# initializing weights
w_0 = rand(X_train.shape[0]) * 2 - 1

# running sd an newton
w_sd, f_sd = steepest_decent(X_train, y_train, w_0, max_iter, epsilon, alph)
w_n, f_n = newton(X_train, y_train, w_0, max_iter, epsilon)

# computing sd accuracy
print(r"Steepest Decent Accuracy:\\")
print(r"Train:", accuracy(X_train, y_train, w_sd), r"\\")
print(r"Test:", accuracy(X_test, y_test, w_sd), r"\\")

# computing newton accuracy
print(r"Newton Accuracy:\\")
print(r"Train:", accuracy(X_train, y_train, w_n), r"\\")
print(r"Test:", accuracy(X_test, y_test, w_n), r"\\")

# plotting convergence history
plt.figure()
plt.plot(f_sd, label="Steepest Decent")
plt.plot(f_n, label="Newton")
plt.legend()
plt.title("8/9 Logistic Regression Convergence History")
plt.ylabel("Logistic Regression Objective")
plt.xlabel("Iteration")

plt.savefig("myplot7.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot7.pdf}")
