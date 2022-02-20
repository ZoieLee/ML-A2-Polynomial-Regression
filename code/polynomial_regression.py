#!/usr/bin/env python

import assignment2 as a2
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a2.load_unicef_data()

np_values = np.asarray(values)
# CMR_1990 = np_values[:,0]
# max_CMR_1990 = np.max(CMR_1990)
# max_country = np.argmax(CMR_1990)
# print(max_CMR_1990, countries[max_country])

# CMR_2011 = np_values[:,1]
# max_CMR_2011 = np.max(CMR_2011)
# max_country = np.argmax(CMR_2011)
# print(max_CMR_2011, countries[max_country])


targets = np_values[:,1]
x = np_values[:,7:]
x = a2.normalize_data(x)


N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


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
    features = np.hstack([x**i for i in range(1, order+1)])
    ones = np.ones((np.size(features, 0), 1), dtype = float)
    features = np.column_stack((ones, features))
    return features


def plot_losses(losses, label = 'loss', color = 'g'):
    x = np.arange(0, len(losses)) + 1
    x[0] = 1
    plt.plot(x, losses, color = color, label = label)
    # plt.semilogy()
    plt.legend()
    plt.xticks(np.arange(1, len(losses)+1, 1))



# train_err = {}
# test_err = {}

train_err = []
test_err = []

for degree in range(1, 9):
    features_train = polynomial_features(x_train, degree)
    w = least_squares(features_train, t_train)
    print(w)
    # print(features_train)
    train_loss = rms_loss(features_train, t_train, w)
    # train_err[degree] = train_loss
    train_err.append(train_loss)

    features_test = polynomial_features(x_test, degree)
    test_loss = rms_loss(features_test, t_test, w)
    # test_err[degree] = test_loss
    test_err.append(test_loss)

print(train_err)
print(test_err)


# Produce a plot of results.
# plt.plot(list(train_err.keys()), list(train_err.values()))
# plt.plot(list(test_err.keys()), list(test_err.values()))
plot_losses(train_err, label = 'Training error', color = 'b')
plot_losses(test_err, label = 'Test error', color = 'r')
plt.xlabel('Polynomial degree')
plt.ylabel('RMS')
# plt.semilogy()
plt.legend(['Training error','Test error'])
# plt.title('Fit with polynomials, no regularization')
plt.title('Fit with polynomials, regularized')
plt.show()
