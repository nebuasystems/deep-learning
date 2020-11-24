import numpy as np

# sigmoid activation function
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

# relu activation function
def relu(x):
    # if x > 0:
    #     return x
    # else:
    #     return 0

    #return x if x > 0 else 0

    return np.maximum(0, x)

# identity activation function : 항등

def identity(x):
    return x

# softmax activation function : 큰 값에서 nan 반환하는 불안한 함수
def softmax_overflow(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def softmax(x): #오버플로우 문제 대응
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x))