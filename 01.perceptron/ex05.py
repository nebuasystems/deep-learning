# Multi layer perceptron

import sys
import os
from pathlib import Path
import numpy as np

try:
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(Path(os.getcwd()), 'lib'))
    from common import identity
    from ex02 import AND
    from ex03 import NAND
    from ex04 import OR
except ImportError:
    print('lib ??')


def XOR(x):
    a1 = NAND(x)
    a2 = OR(x)
    a3 = AND(np.array([a1, a2]))

    y = identity(a3)    #출력함수 적용 지점 : q

    return y

if __name__ == '__main__':      #본 파일이 메인일 경우에만 실행
    y1 = XOR(np.array([0, 0]))
    print(y1)

    y2 = XOR(np.array([0, 1]))
    print(y2)
    y3 = XOR(np.array([1, 0]))
    print(y3)

    y4 = XOR(np.array([1, 1]))
    print(y4)