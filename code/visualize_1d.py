#!/usr/bin/env python

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a2.load_unicef_data()

np_values = np.asarray(values)
targets = np_values[:,1]
x = np_values[:,7:]
# x = a2.normalize_data(x)

N_TRAIN = 100;
# Select a single feature.
x_train = x[0:N_TRAIN,3]
# x_train = x[0:N_TRAIN,4]
# x_train = x[0:N_TRAIN,5]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:,3]
# x_test = x[N_TRAIN:,4]
# x_test = x[N_TRAIN:,5]
t_test = targets[N_TRAIN:]
degree = 3

def least_squares(x, y):
    # xTx = x.T.dot(x)
    # xTx_inv = np.linalg.inv(xTx)
    # w = xTx_inv.dot(x.T.dot(y))
    w = np.linalg.pinv(x).dot(y)
    return w

def rms_loss(x, y, w):
    y_hat = x.dot(w)
    loss = np.sqrt(np.mean((y_hat - y) ** 2))
    return loss


def polynomial_features(x, order):
    features = np.column_stack([x**i for i in range(1, order+1)])
    ones = np.ones((np.size(features, 0), 1), dtype = float)
    features = np.column_stack((ones, features))
    return features


# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
# print(min(x_train), max(x_train))

features = polynomial_features(x_train, degree)
w = least_squares(features, t_train)

x_ev_features = polynomial_features(x_ev, degree)

# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
y_ev = np.random.random_sample(x_ev.shape)
y_ev = x_ev_features.dot(w)

print(x_ev_features.shape)
print(w.shape)
print(y_ev.shape)

plt.plot(x_train,t_train,'bo')
plt.plot(x_test,t_test,'go')
plt.legend(['training data','test data'])
plt.plot(x_ev,y_ev,'r.-')
plt.title('A visualization of a regression estimate using random outputs (feature 11)')
# plt.title('A visualization of a regression estimate using random outputs (feature 12)')
# plt.title('A visualization of a regression estimate using random outputs (feature 13)')
plt.show()
