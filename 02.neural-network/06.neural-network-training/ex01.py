# 신경망 학습: 오차제곱합(Sum of Sqruare Error, SSE)

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sum_square_error
except ImportError:
    print('lib ??')

t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])

y1 = np.array([0.1, 0.05, 0.7, 0., 0.02, 0.03, 0.1, 0., 0., 0.])
y2 = np.array([0.1, 0.05, 0., 0.6, 0.02, 0.03, 0.1, 0.1, 0., 0.])
y3 = np.array([0., 0.92, 0.02, 0., 0.03, 0.03, 0., 0., 0., 0.])

# print(np.sum(y1))
# print(np.sum(y2))
# print(np.sum(y3))

#test
print(sum_square_error(y1, t))
print(sum_square_error(y2, t))
print(sum_square_error(y3, t))