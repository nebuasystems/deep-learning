# 계단 함수

import sys
import os
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import step
except ImportError:
    print('lib ??')

x = np.arange(-5.0, 5.0, 0.1)
y = step(x)
print(y)


plt.plot(x, y)
plt.show()