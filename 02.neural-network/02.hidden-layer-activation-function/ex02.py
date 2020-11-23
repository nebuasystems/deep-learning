# relu function & graph
import sys
import os
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))

    from common import relu
except ImportError:
    print('lib ??')

x = np.arange(-10, 10, 0.1)
y = relu(x)

plt.plot(x, y)
plt.show()