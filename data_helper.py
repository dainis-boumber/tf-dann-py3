import numpy as np


def get_data(name):
    Xy = np.genfromtxt('data/' + name + '.csv', delimiter=',')

    X = Xy[:, :-1]
    y = Xy[:, -1]

    return X, y
