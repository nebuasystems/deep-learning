# 신경망학습: 신경망에서의 기울기

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    from common import cross_entropy_error, softmax, numerical_gradient2
except ImportError:
    print('lib ??')

x = np.array([0.6, 0.9])       # input(x)          2 vector
t = np.array([0., 0., 1.])     # label(one-hot)    3 vector

params = {
    'w1': np.array([[0.02, 0.224, 0.135],  [0.01, 0.052, 0.345]]),
    'b1': np.array([0.45, 0.23, 0.11])
}


def forward_propagation():
    w1 = params['w1']
    b1 = params['b1']

    a = np.dot(x, w1) + b1
    y = softmax(a)

    return y


def loss():
    y = forward_propagation()
    e = cross_entropy_error(y, t)

    return e

def numerical_gradient_net():
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_gradient = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:      #각 매게변수 w에 대한 편미분
            idx = it.multi_index
            temp = param[idx]

            param[idx] = temp + h
            h1 = loss()

            param[idx] = temp - h
            h2 = loss()

            param_gradient[idx] = (h1 - h2) / (2 * h)
            param[idx] = temp

            it.iternext()

            gradient[key] = param_gradient

    return gradient


g = numerical_gradient_net()
print(g)

