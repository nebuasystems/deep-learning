# MNIST 손글씨 숫자 분류 신경망(Neural Network for MNIST Handwritten Digit Classification): 전체 시험(test)

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

try:
    #sys.path.append(os.path.join(os.getcwd()))
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist, init_network
    from common import sigmoid, softmax, identity
except ImportError:
    print('lib ??')


# 1. 매개변수(w, b) 데이터 셋 가져오기
network = init_network()
w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

# print(w1.shape)     # 784 * 50 matrix
# print(w2.shape)     # 50 * 100 matrix
# print(w3.shape)     # 100 * 10 matrix
#
# print(b1.shape)     # 50 vector
# print(b2.shape)     # 100 vector
# print(b3.shape)     # 10 vector


# 2. 학습 시험 데이터 가져오기
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

xlen = len(test_x)

# 3. 테스트
for idx in range(xlen):

    x = test_x[idx]

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    predict = np.argmax(y)
    t = test_t[idx]

    print(f'test image index #{idx+1}, max:{np.max(y)}, predict:{predict}, label:{t}')
