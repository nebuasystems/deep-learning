# Training Neural Nework
# Data Set : MNIST Handwritten Digit Dataset
# Network: TwolayerNet
# Estimation: Training
# pkl 파일을 읽어 보자고. 학습이 잘되었는지.

import os
import pickle
from matplotlib import pyplot as plt

train_loss_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_train_losses.pkl')
#train_loss_file = os.path.join(os.getcwd(), 'dataset', 'sample_weight.pkl')
#train_loss_file = os.path.join(os.getcwd(), 'dataset', 'twolayer_params.pkl')
train_losses = None

with open(train_loss_file, 'rb') as f:
    train_losses = pickle.load(f)

print(train_losses)

plt.plot(train_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()