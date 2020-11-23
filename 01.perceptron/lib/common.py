import numpy as np


# step 활성
def step(x):
    return np.array(x > 0, dtype=np.int)

# identity 함수

def identity(x):
    return x
