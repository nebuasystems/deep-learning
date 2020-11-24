# 출력함수(출력층 활성함수) σ() - 항등함수(Identity Function)

import sys
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(os.getcwd()))
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import identity, sigmoid
except ImportError:
    print('lib ??')

# identity activation function
x = np.arange(-10, 10, 0.1)
y = identity(x)





