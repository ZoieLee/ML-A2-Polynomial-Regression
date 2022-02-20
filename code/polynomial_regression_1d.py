#!/usr/bin/env python

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a2.load_unicef_data()

np_values = np.array(values)

targets = np_values[:,1]
x = np_values[:,7:15]
# x = a2.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

print(x_train)


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
    print("poly features: " + str(features.shape))
    ones = np.ones((np.size(features, 0), 1), dtype = float)
    features = np.column_stack((ones, features))
    return features


def plot_losses(losses, label = 'loss'):
    bar_width = 0.3
    x = np.arange(0, len(losses)) + 1
    x[0] = 1
    plt.bar(x, losses, label = label)
    plt.semilogy()
    plt.legend()
    plt.xticks(np.arange(1, len(losses)+1, 1))



train_err = {}
test_err = {}
degree = 3

for feature_index in range(0, np.size(x, axis=1)):
    x_train_index = x_train[:, feature_index]
    features_train = polynomial_features(x_train_index, degree)
    print("train features: "+ str(x_train_index.shape))

    w = least_squares(features_train, t_train)
    train_loss = rms_loss(features_train, t_train, w)
    train_err[feature_index + 8] = train_loss

    x_test_index = x_test[:, feature_index]
    features_test = polynomial_features(x_test_index, degree)
    test_loss = rms_loss(features_test, t_test, w)
    test_err[feature_index + 8] = test_loss

print(train_err)
print(test_err)



# Produce a plot of results.
plt.bar([i - 0.2 for i in train_err.keys()], height = train_err.values(), width = 0.4)
plt.bar([i + 0.2 for i in test_err.keys()], height = test_err.values(), width = 0.4)
plt.xlabel('Polynomial degree')
plt.ylabel('RMS')
# plt.semilogy()
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, no regularization')
# plt.title('Fit with polynomials, regularized')
plt.show()
