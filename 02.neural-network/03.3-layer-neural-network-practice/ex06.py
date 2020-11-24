# 3층 신경망 신호 전달 구현6 : 출력층 출력함수identity σ() 적용
import sys
import os
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(os.getcwd()))
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import identity
    from ex05 import a3
except ImportError:
    print('lib ??')

print('\n= 신호 전달 구현6: 출력층 출력함수identitu σ() 적용=================================')
print(f'a3 dimension: {a3.shape}')  # 2 vector

y = identity(a3)
print(f'y = {y}')
