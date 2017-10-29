import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from reference_perceptron import *

REG = 0.00001
C = 10.
MIN_EPOCH = 100
MAX_EPOCH = 125

def perceptron_train(xt, yt, xd, yd):
    epoch = 0
    w = np.array([0. for t in xt[0]])
    best_err = float('inf')
    best_w = w
    while epoch < MAX_EPOCH:
        idxs = [i for i in range( len( xt ) )]
        random.shuffle( idxs )
        for i in idxs:
            if yt[i] * np.dot(w, xt[i]) <= 0.:
                w += yt[i] * xt[i]
        err = sum([int(yd[i] * np.dot(w, xd[i]) <= 0.) for i in range(len(xd))])
        err_rate = err / float(len(xd))
        if err_rate < best_err:
            best_err = err_rate
            best_w = w
        else:
            w = best_w
        print 'epoch {}\t\tpereptron error rate: {}'.format(epoch, err_rate)
        epoch += 1
    return best_w, best_err


def pegasos_train(xt, yt, xd, yd, c, target_delta):
    total_svs = 0
    l = 2. / (len( xd ) * c)
    best_err_rate = float('inf')
    best_w, best_epoch = None, None
    epoch = 0
    objs, tr_errs, d_errs = [], [], []
    while epoch < MAX_EPOCH:
        idxs = [i for i in range( len( xt ) )]
        random.shuffle( idxs )
        x_shuffle, y_shuffle = [xt[i] for i in idxs], [yt[i] for i in idxs]
        w, sv_count = train_epoch(x_shuffle, y_shuffle, epoch, c, best_w)
        # training and dev error
        t_err = test(xt, yt, w)
        err = test(xd, yd, w)
        # objective fn
        """
        obj = 1. / len(xd) * sum([
            l / 2 * np.dot(w, w) + max(0, 1. - yd[i] * np.dot(w, xd[i]))
            for i in range(len(xd))
        ])
        """
        obj = l / 2. * np.dot(w, w) + 1. / len(xt) * sum([max(0., 1. - yt[i] * np.dot(w, xt[i])) 
            for i in range(len(xt))])
        print '{}\t{}\t{}\t{}'.format( epoch, obj, t_err, err )
        objs.append(obj)
        tr_errs.append(t_err)
        d_errs.append(err)
        if err < best_err_rate:
            best_w = w
            best_epoch = epoch
            total_svs += sv_count
            if best_err_rate - err < target_delta and epoch > MIN_EPOCH:
                print '\nsupport vectors: {}'.format(sv_count)
                return best_w, err, best_epoch, objs, tr_errs, d_errs
            best_err_rate = err
        epoch += 1

    print 'support vectors: {}'.format(sv_count)

    return best_w, best_err_rate, best_epoch, objs, tr_errs, d_errs
            

def train_epoch(x, y, epoch, c, w=None):
    sv_count = 0
    if w is None:
        w = np.array([0. for i in range(len(x[0]))])
    l = 2. / (len(x) * c)
    for t in range(len(x)):
        learn_rate = 1. / (l * (epoch * len(x) + t + 1))
        if y[t] * np.dot( w, x[t] ) < 1.:
            w = w - learn_rate * (l * w - y[t] * x[t])
            sv_count += 1
        else:
            w = w - learn_rate * l * w
    return w, sv_count


def predict(x, w):
    return [np.dot(w, x_i) for x_i in x]


def test(x, y, w):
    p = predict(x, w)
    return sum([y[i] * p[i] <= 0. for i in range(len(y))]) / float(len(y))


def get_data(train_file, dev_file):
    feature2index = create_feature_map(train_file)
    train_data = map_data(train_file, feature2index)
    dev_data = map_data(dev_file, feature2index)
    test_data = map_data( 'income-data/income.test.txt', feature2index)
    return feature2index, train_data, dev_data, test_data


def main():
    train_file, dev_file = "income-data/income.train.txt.5k", \
                           "income-data/income.dev.txt"
    f2i, dt, dd, dtest = get_data(train_file, dev_file)
    xt, yt = [d[0] for d in dt], [d[1] for d in dt]
    xd, yd = [d[0] for d in dd], [d[1] for d in dd]
    weights, err_rate, best_epoch, objs, tr_errs, d_errs = pegasos_train(xt, yt, xd, yd, 1., 0.001)


    # print predictions
    test_lines = [l for l in open('income-data/income.test.txt', 'r')]
    positive = 0
    for i in range(len(dtest)):
        pred = np.dot( dtest[i][0], weights )
        print >> sys.stderr, test_lines[i].strip(),
        if pred <= 0.:
            print >> sys.stderr, '<=50K'
        else:
            print >> sys.stderr, '>50K'
        # print >> sys.stderr, pred > 0.
        if pred > 0.:
            positive += 1
    print '\n\nbest err rate: {} achieved in epoch {}'.format( err_rate, best_epoch )
    print 'positive frac: {}'.format(positive / float(len(dtest)))

    plt.plot(range(len(objs)), objs)
    plt.title('Objective Function vs. Epoch')
    plt.show()

    trplt = plt.plot(range(len(tr_errs)), tr_errs)
    plt.title('Error vs. Epoch')
    dplt = plt.plot(range(len(d_errs)), d_errs)
    plt.legend( ('Training Error', 'Dev Error'), numpoints = 1)
    plt.show()

if __name__ == '__main__':
    main()
