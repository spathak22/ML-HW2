import numpy as np
import random

def pegasos_train(x, y):
    pass

def train_epoch(x, y, reg_const, epoch_length, w=None):
    if w is None:
        w = np.array([0. for i in range(len(x[0]))])
    for t in range(epoch_length):
        learn_rate = 1. / (reg_const * (t + 1))
        i = random.randint(0, len(x) - 1)
        if y[i] * np.dot(w, x[i]) < 1.:
            w = (1. - reg_const * learn_rate) * w + learn_rate * y[i] * x[i]
        else:
            w = (1. - reg_const * learn_rate) * w
    return w

x = [np.array([1., -1.]), np.array([-1., 1.])]
y = [1., -1.]
print train_epoch(x, y, 0.0001, 2)