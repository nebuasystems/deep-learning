# 신경망학습: 신경망에서의 기울기
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax, sigmoid, cross_entropy_error, relu
except ImportError:
    print('Library Module Can Not Found')


params = dict()
            #   784         50         10
def initialize(sz_input, sz_hidden, sz_output, w_init=0.01):
    params['w1'] = w_init * np.random.randn(sz_input, sz_hidden)    # (784, 50)
    params['b1'] = np.zeros(sz_hidden)                              # (50,)
    params['w2'] = w_init * np.random.randn(sz_hidden, sz_output)   # (50, 10)
    params['b2'] = np.zeros(sz_output)                              # (10,)


def forward_progation(x):
    w1 = params['w1']
    b1 = params['b1']           #   data수 * 입력수,     입력수 * 출력수,    출력수
    a1 = np.dot(x, w1) + b1     # x: 100  *   784,  w1: 784   * 50,    b1: 50

    z1 = sigmoid(a1)            #역전파시에는 sigmoid 사용하면 안됨
    #z1 = relu(a1)

    w2 = params['w2']
    b2 = params['b2']           #   data수 * 입력수,     입력수 * 출력수,    출력수
    a2 = np.dot(z1, w2) + b2    # z1: 100  *   50,  w2: 50   * 10,    b2: 10
                                # data수  * 출력수
    y = softmax(a2)             # y: 100    * 10

    return y


def loss(x, t):
    y = forward_progation(x)
    e = cross_entropy_error(y, t)
    return e


def accuracy(x, t):
    y = forward_progation(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    acc = np.sum(y == t) / float(x.shape[0])

    return acc


def numerical_gradient_net(x, t):

    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]                     # 전체 파라메터에서 한 종(key)만 가져 온다. 참조 방식?
        param_gradient = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:                  # 해당(key) 파라메터들을 편미분하여 기울기 행렬을 만든다.
            idx = it.multi_index
            temp = param[idx]

            param[idx] = temp + h
            h1 = loss(x, t)                             # scalar

            param[idx] = temp - h
            h2 = loss(x, t)                             # scalar

            param_gradient[idx] = (h1 - h2) / (2 * h)   # scalar

            param[idx] = temp   # 값복원
            it.iternext()

        gradient[key] = param_gradient                  # 해당(key) 파라메터에 대한 기울기 행렬

    return gradient                                     # 모든 파라메터들에 대한 기울기 행렬


