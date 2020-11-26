# 신경망학습: 교차 엔트로피 손실함수(Cross Entropy Err, CEE)

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import cross_entropy_error_non_batch
except ImportError:
    print('lib ??')

t = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])

y1 = np.array([0.1, 0.05, 0.7, 0., 0.02, 0.03, 0.1, 0., 0., 0.])
y2 = np.array([0.1, 0.05, 0., 0.6, 0.02, 0.03, 0.1, 0.1, 0., 0.])
y3 = np.array([0., 0.92, 0.02, 0., 0.03, 0.03, 0., 0., 0., 0.])

#test
print(cross_entropy_error_non_batch(y1, t))
print(cross_entropy_error_non_batch(y2, t))
print(cross_entropy_error_non_batch(y3, t))