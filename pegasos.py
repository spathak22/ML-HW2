import numpy as np
import random
import sys
from reference_perceptron import *

REG = 0.00001
C = 10.
MAX_EPOCH = 200

def pegasos_train(xt, yt, xd, yd, c, target_delta):
    best_err_rate = float('inf')
    best_w = None
    epoch = 0
    while epoch < MAX_EPOCH:
        idxs = [i for i in range( len( xt ) )]
        # random.shuffle( idxs )
        x_shuffle, y_shuffle = [xt[i] for i in idxs], [yt[i] for i in idxs]
        w = train_epoch(x_shuffle, y_shuffle, epoch, c, best_w)
        err = test(xd, yd, w)
        
        if err < best_err_rate:
            print >> sys.stderr, 'epoch {}: dev err {}'.format(epoch, err)
            best_w = w
            if best_err_rate - err < target_delta:
                return best_w, best_err_rate
            best_err_rate = err
        epoch += 1

    return best_w, best_err_rate
            

def train_epoch(x, y, epoch, c, w=None):
    if w is None:
        w = np.array([0. for i in range(len(x[0]))])
    # "regularization" constant
    l = 2. / (len(x) * c)
    # train for this epoch
    for t in range(len(x)):
        learn_rate = 1. / (l * (epoch*len(x) + t + 1))
        if y[t] * np.dot( w, x[t] ) < 1.:
            w = w * (1. - learn_rate * l) + learn_rate * y[t] * x[t]
        else:
            w = w * (1. - learn_rate * l)
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
    for k in range(-2, 3, 1):
        c = 10.**k
        weights, err_rate = pegasos_train(xt, yt, xd, yd, c, 0.0001)
        print 'C = {}\t\tbest err rate: {}'.format(c, err_rate)

if __name__ == '__main__':
    main()
