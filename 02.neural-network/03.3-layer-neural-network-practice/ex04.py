# 3층 신경망 신호 전달 구현4 : 은닉 2층 활성함수 h() 적용
import sys
import os
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt

try:
    sys.path.append(os.path.join(os.getcwd()))
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import sigmoid
    from ex03 import a2
except ImportError:
    print('lib ??')

print('\n= 신호 전달 구현4: 은닉 2층 활성함수sigmoid h() 적용=================================')
print(f'a2 dimension: {a2.shape}')  # 3 vector

z2 = sigmoid(a2)
print(f'z2 = {z2}')

