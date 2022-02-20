#!/usr/bin/env python

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

(countries, features, values) = a2.load_unicef_data()

np_values = np.asarray(values)

targets = np_values[:,1]
x = np_values[:,7:]
x = a2.normalize_data(x)


N_TRAIN = 100;
x_ = x[0:N_TRAIN,:]
t_ = targets[0:N_TRAIN]

lambs = np.array([0, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5])
degree = 2
batch = 10

def ridge_reg(x, y, l):
    xTx = x.T.dot(x)
    i = np.identity(np.size(xTx, 1))
    inv = np.linalg.inv(xTx + l * i)
    w = inv.dot(x.T.dot(y))
    return w

def rms_loss_l2(x, y, w, l):
    y_hat = x.dot(w)
    # loss = np.sqrt(np.mean((y_hat - y) ** 2 + l * np.sum(w ** 2)))
    loss = np.sqrt(np.mean((y_hat - y) ** 2))
    return loss

def polynomial_features(x, order):
    features = np.hstack([x**i for i in range(1, order+1)])
    ones = np.ones((np.size(features, 0), 1), dtype = float)
    features = np.column_stack((ones, features))
    return features


vali_error = []

for l in lambs:
    sum_error = []
    for i in range(0, 10):
        a = i * batch
        b = (i + 1) * batch
        x_vali = x_[a:b, :]
        t_vali = t_[a:b]
        x_train = np.delete(x_, slice(a, b), axis = 0)
        t_train = np.delete(t_, slice(a, b), axis = 0)
        print("when lambda = " + str(l) + ", batch = " + str(i))
        print(x_train.shape)
        print(t_train.shape)

        features_train = polynomial_features(x_train, degree)
        print(features_train.shape)
        w = ridge_reg(features_train, t_train, l)
        print(w.shape)
        features_vali = polynomial_features(x_vali, degree)
        error = rms_loss_l2(features_vali, t_vali, w, l)
        sum_error.append(error)
    avg = np.average(sum_error)
    vali_error.append(avg)

print(vali_error)
plt.semilogx(lambs[1:], vali_error[1:])
plt.plot([0, vali_error[0]], [0.01, vali_error[0]])
plt.legend(['Validation loss when lambda from 0.01 to 100000','Validation loss when lambda is 0'])
# plt.semilogy()
plt.show()

