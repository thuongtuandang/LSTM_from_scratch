import numpy as np


def softmax(y):
    exp_y = np.exp(y-np.max(y))
    return exp_y/np.sum(exp_y, axis = 0, keepdims=True)

def hadamard(A,B):
    m, n = A.shape
    c = np.zeros_like(A)
    for i in range(m):
        for j in range(n):
            c[i][j] = A[i][j] * B[i][j]
    return c

def sigmoid(x):
    x_clip = np.clip(x, 1e-7, 1 - 1e-7)
    exp_x = np.exp(-x_clip)
    return 1/(1 + exp_x)

def oneHotEncode(index, vocab_size):
    v = np.zeros(vocab_size, dtype=np.int8)
    v[index] = np.int8(1)
    return v