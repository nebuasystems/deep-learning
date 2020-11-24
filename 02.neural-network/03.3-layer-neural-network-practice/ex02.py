# 3층 신경망 신호 전달 구현2 : 은닉 1층 활성함수 h() 적용
import sys
import os
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(os.getcwd()))
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid
    from ex01 import a1
except ImportError:
    print('lib ??')

print('\n= 신호 전달 구현1: 은닉 1층 활성함수sigmoid h() 적용=================================')
print(f'a1 dimension: {a1.shape}')  # 3 vector

z1 = sigmoid(a1)
print(f'z1 = {z1}')

