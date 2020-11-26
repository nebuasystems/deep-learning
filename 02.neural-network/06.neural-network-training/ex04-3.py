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


def forward_propagation(w):
    a = np.dot(x, w)
    y = softmax(a)

    return y


def loss(w):
    y = forward_propagation(w)
    e = cross_entropy_error(y, t)

    return e


_w = np.array([
    [0.02, 0.224, 0.135],
    [0.01, 0.052, 0.345]
])                              # weight            2 * 3 matrix

g = numerical_gradient2(loss, _w)
print(g)

