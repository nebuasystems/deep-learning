# 출력함수(출력층 활성함수) σ() - softmax Function

import sys
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import softmax
except ImportError:
    print('lib ??')

# test1
a = np.array([0.3, 1., 0.78])
y = softmax(a)
print(y, np.sum(y))

# test2: 큰 값
a = np.array([0.3, 800., 0.78])
y = softmax(a)
print(y, np.sum(y))

