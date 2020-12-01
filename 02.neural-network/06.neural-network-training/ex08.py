# Training Neural Nework
# Data Set : MNIST Handwritten Digit Dataset
# Network: TwolayerNet
# Estimation: Training Accuracy
# 학습 결과로 얻어진 파라메터들을 시험해 보자. 잘되는지.

import os
import pickle
import sys
import time

import numpy as np
from pathlib import Path

try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from mnist import load_mnist
    import twolayernet as network
except ImportError:
    print('Library Module Can Not Found')

# 1. load train/test data
(train_x, train_t), (test_x, test_t) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 2. load params dataset trained
params_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_params.pkl')
with open(params_file, 'rb') as f:
    network.params = pickle.load(f)

# print(network.params)

train_accuracy = network.accuracy(train_x, train_t)
test_accuracy = network.accuracy(test_x, test_t)

print(train_accuracy, test_accuracy)

# train_accruracy와 test_accuracy가 일치하는 것은 Overfitting이 발생하지 않은 것.
# 학습 중에 1epoch당 train/test accuracy를 측정하여 비교한다.
# 두 accuracy가 마지막까지 차이가 없는 것이 가장 바람직.
# 차이가 나면 그 시점을 파악하여 학습을 중지해야 한다. -> early stop

