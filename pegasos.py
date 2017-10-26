import numpy as np
import random
import sys
from reference_perceptron import *

REG = 0.0001

def pegasos_train(xt, yt, xd, yd, target_delta):
    best_err_rate = float('inf')
    best_w = None
    epoch = 0
    while True:
        w = train_epoch(xt, yt, REG, len(xt), best_w)
        err = test(xd, yd, w)
        
        if err < best_err_rate:
            print >> sys.stderr, 'epoch {}: dev err {}'.format(epoch, err)
            best_w = w
            if best_err_rate - err < target_delta:
                return best_w, best_err_rate
            best_err_rate = err
        epoch += 1
            


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


def predict(x, w):
    return [np.dot(w, x_i) for x_i in x]


def test(x, y, w):
    p = predict(x, w)
    return sum([y[i] * p[i] <= 0. for i in range(len(y))]) / float(len(y))


def get_data(train_file, dev_file):
    feature2index = create_feature_map(train_file)
    train_data = map_data(train_file, feature2index)
    dev_data = map_data(dev_file, feature2index)
    return feature2index, train_data, dev_data


def main():
    train_file, dev_file = "income-data/income.train.txt.5k", \
                           "income-data/income.dev.txt"
    f2i, dt, dd = get_data(train_file, dev_file)
    xt, yt = [d[0] for d in dt], [d[1] for d in dt]
    xd, yd = [d[0] for d in dt], [d[1] for d in dd]
    weights, err_rate = pegasos_train(xt, yt, xd, yd, 0.0001)
    print 'best err rate: {}'.format(err_rate)

if __name__ == '__main__':
    main()
