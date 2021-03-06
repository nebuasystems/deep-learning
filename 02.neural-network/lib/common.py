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

def softmax(x): #오버플로우 문제 & 배치처리 지원 대응
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    y = np.exp(x) / np.sum(np.exp(x))
    return y

# Sum of Square Error(SSE)
def sum_square_error(y, t):
    e = 0.5 * np.sum((y-t)**2)
    return e

# cross entropy error
# t = one hot
 # non batch
def cross_entropy_error_non_batch(y, t):
    delta = 1.e-7
    e = -np.sum(t * np.log(y+delta))
    return e

# cross entropy error
# t = one hot
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]

    delta = 1.e-7
    e = -np.sum(t * np.log(y+delta)) / batch_size       # scalar
    return e

def numerical_diff1(f, w, x, t):

    h = 1e-4
    gradient = np.zeros_like(w)

    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:      #각 매게변수 w에 대한 편미분
        idx = it.multi_index
        temp = w[idx]

        w[idx] = temp + h
        h1 = f(w, x, t)

        w[idx] = temp - h
        h2 = f(w, x, t)

        gradient[idx] = (h1 - h2) / (2 * h)
        w[idx] = temp

        it.iternext()

    return gradient

numerical_gradient1 = numerical_diff1

def numerical_diff2(f, w):

    h = 1e-4
    gradient = np.zeros_like(w)

    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:      #각 매게변수 w에 대한 편미분
        idx = it.multi_index
        temp = w[idx]

        w[idx] = temp + h
        h1 = f(w)

        w[idx] = temp - h
        h2 = f(w)

        gradient[idx] = (h1 - h2) / (2 * h)
        w[idx] = temp

        it.iternext()

    return gradient

numerical_gradient2 = numerical_diff2