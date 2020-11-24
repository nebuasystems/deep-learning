#이전 모듈들을 합체

import sys
import os
from pathlib import Path
import numpy as np

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import identity, sigmoid
except ImportError:
    print('lib ??')

def init_network():
    params = {
        'w1' : np.array([[0.1, 0.2, 0.5], [0.3, 0.4, 1.]]),
        'b1' : np.array([0.1, 0.2, 0.3]),
        'w2' : np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        'b2' : np.array([0.1, 0.2]),
        'w3' : np.array([[0.1, 0.2], [0.3, 0.4]]),
        'b3' : np.array([0.1, 0.2])
    }
    return params


def forward_propagation(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y = identity(a3)

    return y



network = init_network()

x = np.array([1., 5.])
y = forward_propagation(network, x)
print(f'y = {y}')
