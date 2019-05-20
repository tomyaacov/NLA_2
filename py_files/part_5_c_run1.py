from numpy import asarray
from numpy.random import rand
from mnist import MNIST
import matplotlib.pyplot as plt
from py_files.part_5_c import steepest_decent, newton, accuracy
from py_files.part_5_a_helper import reg_obj

# loading data
mndata = MNIST('py_files/data')
mndata.gz = True
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()

# parameters definition
max_iter = 1000
epsilon = 1e-4
alpha = 1

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
w_sd = steepest_decent(X_train, y_train, w_0, max_iter, alpha)
w_n = newton(X_train, y_train, w_0, max_iter, epsilon)

# computing sd accuracy
print(r"0,1:\\")
print(r"Steepest Decent Accuracy:\\")
print(r"Train:", accuracy(X_train, y_train, w_sd[-1]), r"\\")
print(r"Test:", accuracy(X_test, y_test, w_sd[-1]), r"\\")

# computing convergence history - Steepest Decent
ch_sd_train = []
ch_sd_test = []
for w in w_sd:
    obj = reg_obj(X_train, y_train, w)
    ch_sd_train.append(obj)
    obj = reg_obj(X_test, y_test, w)
    ch_sd_test.append(obj)

# plotting convergence history - Steepest Decent
plt.figure()
plt.semilogy([abs(x-ch_sd_train[-1]) for x in ch_sd_train], label="Train")
plt.semilogy([abs(x-ch_sd_test[-1]) for x in ch_sd_test], label="Test")
plt.legend()
plt.title("0/1 Logistic Regression Convergence History - Steepest Decent")
plt.ylabel("LR Objective - Optimal LR Objective")
plt.xlabel("Iteration")

plt.savefig("myplot6.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot6.pdf}")

# computing newton accuracy
print(r"Newton Accuracy:\\")
print(r"Train:", accuracy(X_train, y_train, w_n[-1]), r"\\")
print(r"Test:", accuracy(X_test, y_test, w_n[-1]), r"\\")

# computing convergence history - newton
ch_n_train = []
ch_n_test = []
for w in w_n:
    obj = reg_obj(X_train, y_train, w)
    ch_n_train.append(obj)
    obj = reg_obj(X_test, y_test, w)
    ch_n_test.append(obj)

# plotting convergence history - Steepest Decent
plt.figure()
plt.semilogy([abs(x-ch_n_train[-1]) for x in ch_n_train], label="Train")
plt.semilogy([abs(x-ch_n_test[-1]) for x in ch_n_test], label="Test")
plt.legend()
plt.title("0/1 Logistic Regression Convergence History - Newton")
plt.ylabel("LR Objective - Optimal LR Objective")
plt.xlabel("Iteration")

plt.savefig("myplot7.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot7.pdf}")

# plotting convergence train
plt.figure()
plt.plot(ch_sd_train, label="Steepest Decent")
plt.plot(ch_n_train, label="Newton")
plt.legend()
plt.title("0/1 Logistic Regression Train Set Convergence")
plt.ylabel("Logistic Regression Objective")
plt.xlabel("Iteration")

plt.savefig("myplot8.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot8.pdf}")

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
w_sd = steepest_decent(X_train, y_train, w_0, max_iter, alpha)
w_n = newton(X_train, y_train, w_0, max_iter, epsilon)

# computing sd accuracy
print(r"8,9:\\")
print(r"Steepest Decent Accuracy:\\")
print(r"Train:", accuracy(X_train, y_train, w_sd[-1]), r"\\")
print(r"Test:", accuracy(X_test, y_test, w_sd[-1]), r"\\")

# computing convergence history - Steepest Decent
ch_sd_train = []
ch_sd_test = []
for w in w_sd:
    obj = reg_obj(X_train, y_train, w)
    ch_sd_train.append(obj)
    obj = reg_obj(X_test, y_test, w)
    ch_sd_test.append(obj)

# plotting convergence history - Steepest Decent
plt.figure()
plt.semilogy([abs(x-ch_sd_train[-1]) for x in ch_sd_train], label="Train")
plt.semilogy([abs(x-ch_sd_test[-1]) for x in ch_sd_test], label="Test")
plt.legend()
plt.title("8/9 Logistic Regression Convergence History - Steepest Decent")
plt.ylabel("LR Objective - Optimal LR Objective")
plt.xlabel("Iteration")

plt.savefig("myplot9.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot9.pdf}")

# computing newton accuracy
print(r"Newton Accuracy:\\")
print(r"Train:", accuracy(X_train, y_train, w_n[-1]), r"\\")
print(r"Test:", accuracy(X_test, y_test, w_n[-1]), r"\\")

# computing convergence history - newton
ch_n_train = []
ch_n_test = []
for w in w_n:
    obj = reg_obj(X_train, y_train, w)
    ch_n_train.append(obj)
    obj = reg_obj(X_test, y_test, w)
    ch_n_test.append(obj)

# plotting convergence history - Steepest Decent
plt.figure()
plt.semilogy([abs(x-ch_n_train[-1]) for x in ch_n_train], label="Train")
plt.semilogy([abs(x-ch_n_test[-1]) for x in ch_n_test], label="Test")
plt.legend()
plt.title("8/9 Logistic Regression Convergence History - Newton")
plt.ylabel("LR Objective - Optimal LR Objective")
plt.xlabel("Iteration")

plt.savefig("myplot10.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot10.pdf}")

# plotting convergence train
plt.figure()
plt.plot(ch_sd_train, label="Steepest Decent")
plt.plot(ch_n_train, label="Newton")
plt.legend()
plt.title("8/9 Logistic Regression Train Set Convergence")
plt.ylabel("Logistic Regression Objective")
plt.xlabel("Iteration")

plt.savefig("myplot11.pdf", bbox_inches="tight")
print(r"\saveandshowplot{myplot11.pdf}")
