# Training Neural Nework
# Data Set : MNIST Handwritten Digit Dataset
# Network: TwolayerNet
# Estimation: Training
# pkl 파일을 읽어 보자고. 학습이 잘되었는지.

import os
import pickle
from matplotlib import pyplot as plt

trainacc_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_train_accuracy.pkl')
testacc_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_test_accuracy.pkl')


with open(trainacc_file, 'rb') as f_trainacc, open(testacc_file, 'rb') as f_testacc:
    train_accuracies = pickle.load(f_trainacc)
    test_accuracies = pickle.load(f_testacc)

plt.plot(train_accuracies, label='train_accuracy')
plt.plot(test_accuracies, label='test_accuracy')

plt.xlim(0, 20, 1)
plt.ylim(0., 1., 0.5)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.show()