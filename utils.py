import numpy as np


def softmax(y):
    exp_y = np.exp(y-np.max(y))
    return exp_y/np.sum(exp_y, axis = 0, keepdims=True)