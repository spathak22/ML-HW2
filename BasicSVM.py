# !/usr/bin/env python

from __future__ import division

import sys
import numpy as np
import time
from collections import defaultdict
from sklearn import svm
import matplotlib.pyplot as plt


def plot_error_rates(errorRates, title, legends):
    if len(errorRates) == 1:
        xValues1, yValues1 = zip(*errorRates[0])
        plt.plot(xValues1, yValues1, 'r')
    elif len(errorRates) == 2:
        xValues1, yValues1, = zip(*errorRates[0])
        xValues2, yValues2, = zip(*errorRates[1])
        plt.plot(xValues1, yValues1, 'r', xValues2, yValues2, 'b')
    elif len(errorRates) == 3:
        xValues1, yValues1 = zip(*errorRates[0])
        xValues2, yValues2 = zip(*errorRates[1])
        xValues3, yValues3 = zip(*errorRates[2])
        plt.plot(xValues1, yValues1, 'r', xValues2, yValues2, 'b', xValues3, yValues3, 'g')

    plt.ylabel('Time in Seconds')
    plt.xlabel('C')
    plt.title(title)
    plt.legend(legends)
    plt.show()


def map_data(filename, feature2index):
    data = []  # list of (vecx, y) pairs
    dimension = len(feature2index)
    for j, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(dimension)
        for i, fv in enumerate(features[:-1]):  # last one is target
            if (i, fv) in feature2index:  # ignore unobserved features
                feat_vec[feature2index[i, fv]] = 1
        feat_vec[0] = 1  # bias
        data.append((feat_vec, 1 if features[-1] == ">50K" else -1))

    return data


def train(train_data, dev_data, it=1, MIRA=False, check_freq=5000, aggressive=0.9, verbose=True):
    train_size = len(train_data)
    dimension = len(train_data[0][0])
    model = np.zeros(dimension)
    totmodel = np.zeros(dimension)
    best_err_rate = best_err_rate_avg = best_positive = best_positive_avg = 1
    t = time.time()
    for i in xrange(1, it + 1):
        print "starting epoch", i
        for j, (vecx, y) in enumerate(train_data, 1):
            s = model.dot(vecx)
            if not MIRA:  # perceptron
                if s * y <= 0:
                    model += y * vecx
            else:  # MIRA
                if s * y <= aggressive:
                    model += (y - s) / vecx.dot(vecx) * vecx
            totmodel += model  # stupid!
            if j % check_freq == 0:
                dev_err_rate, positive = test(dev_data, model)
                dev_err_rate_avg, positive_avg = test(dev_data, totmodel)
                epoch_position = i - 1 + j / train_size
                if dev_err_rate < best_err_rate:
                    best_err_rate = dev_err_rate
                    best_err_pos = epoch_position  # (i, j)
                    best_positive = positive
                if dev_err_rate_avg < best_err_rate_avg:
                    best_err_rate_avg = dev_err_rate_avg
                    best_err_pos_avg = epoch_position  # (i, j)
                    best_positive_avg = positive_avg
                    best_avg_model = totmodel

    print "training %d epochs costs %f seconds" % (it, time.time() - t)
    print "MIRA" if MIRA else "perceptron", aggressive if MIRA else "", \
        "unavg err: {:.2%} (+:{:.1%}) at epoch {:.2f}".format(best_err_rate,
                                                              best_positive,
                                                              best_err_pos), \
        "avg err: {:.2%} (+:{:.1%}) at epoch {:.2f}".format(best_err_rate_avg,
                                                            best_positive_avg,
                                                            best_err_pos_avg)

    return best_avg_model


def test(data, model):
    errors = sum(model.dot(vecx) * y <= 0 for vecx, y in data)
    positives = sum(model.dot(vecx) > 0 for vecx, _ in data)  # stupid!
    return errors / len(data), positives / len(data)


def create_feature_map(train_file):
    column_values = defaultdict(set)
    for line in open(train_file):
        line = line.strip()
        features = line.split(", ")[:-1]  # last field is target
        for i, fv in enumerate(features):
            column_values[i].add(fv)

    feature2index = {(-1, 0): 0}  # bias
    for i, values in column_values.iteritems():
        for v in values:
            feature2index[i, v] = len(feature2index)

    dimension = len(feature2index)
    # print "dimensionality: ", dimension
    return feature2index


arg1 = ''
if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]

    train_file, dev_file = "../income-data/income.train.txt.5k", "../income-data/income.dev.txt"

    feature2index = create_feature_map(train_file)
    train_data = map_data(train_file, feature2index)
    dev_data = map_data(dev_file, feature2index)

    # model = train(train_data, dev_data, it=5, MIRA=False, check_freq=5000, verbose=False)
    # print "train_err {:.2%} (+:{:.1%})".format(*test(train_data, model))
    # model = train(train_data, dev_data, it=5, MIRA=True, check_freq=5000, verbose=False, aggressive=.9)
    # print "train_err {:.2%} (+:{:.1%})".format(*test(train_data, model))

# 1.1
if arg1 == '1.1':
    train_X, train_Y = zip(*train_data)
    dev_X, dev_Y = zip(*dev_data)
    train_size = len(train_Y)
    dev_size = len(dev_Y)

    begin_time = time.time()
    clf = svm.SVC(kernel='linear', C=1)

    clf.fit(train_X, train_Y)
    train_err = sum(svm_Y != y for svm_Y, y in zip(clf.predict(train_X), train_Y))
    print "Q1.1 train_err {:.2%} ".format(train_err / train_size)

    dev_err = sum(svm_Y != y for svm_Y, y in zip(clf.predict(dev_X), dev_Y))
    print "Q1.1 dev_err {:.2%} ".format(dev_err / dev_size)

    end_time = time.time()
    print "Q1.1 Training time is ", end_time - begin_time

# 1.2
elif arg1 == '1.2':
    train_X, train_Y = zip(*train_data)
    begin_time = time.time()
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(train_X, train_Y)

    print "Q1.2 There are", clf.n_support_, " support vectors"
    w = clf.coef_[0]
    violation_num = 0
    b = clf.intercept_
    for x, y in train_data:
        if max(0, 1 - (y * (np.inner(x, w) + b))) != 0:
            violation_num += 1

    print "Q1.2 There are", violation_num, "margin violations"
    end_time = time.time()
    print "Time is ", end_time - begin_time

# 1.3
elif arg1 == '1.3':
    train_X, train_Y = zip(*train_data)
    begin_time = time.time()
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(train_X, train_Y)
    w = clf.coef_[0]
    violation_amount = 0
    b = clf.intercept_
    for x, y in train_data:
        k = 1 - (y * (np.inner(x, w) + b))
        if max(0, k) != 0:
            violation_amount += max(0, k)

    print "Q1.3 the total amount of margin violations", violation_amount
    print "Q1.3 the objectives are", (1/(2*len(train_X)) * np.inner(w, w) )+ (violation_amount/len(train_X))
    end_time = time.time()
    print "Time is ", end_time - begin_time

# 1.4
elif arg1 == '1.4':
    begin_time = time.time()

    end_time = time.time()
    print "Time is ", end_time - begin_time

# 1.5
elif arg1 == '1.5':
    train_X, train_Y = zip(*train_data)
    dev_X, dev_Y = zip(*dev_data)
    train_size = len(train_Y)
    dev_size = len(dev_Y)
    c_list = [0.01, 0.1, 1, 2, 5, 10]
    for c in c_list:
        begin_time = time.time()
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(train_X, train_Y)
        train_err = sum(svm_Y != y for svm_Y, y in zip(clf.predict(train_X), train_Y))
        print "Q1.5 C =", c
        print "train_err {:.2%} ".format(train_err / train_size)

        dev_err = sum(svm_Y != y for svm_Y, y in zip(clf.predict(dev_X), dev_Y))
        print "dev_err {:.2%} ".format(dev_err / dev_size)

        print "Support vectors", clf.n_support_
        end_time = time.time()
        print "Training time is ", end_time - begin_time

# 1.7
elif arg1 == '1.7':
    train_X, train_Y = zip(*train_data)
    dev_X, dev_Y = zip(*dev_data)
    train_size = len(train_Y)
    dev_size = len(dev_Y)
    c_list = [0.01, 0.1, 1, 2, 5, 10]
    train_error_rates = []
    dev_error_rates = []
    for c in c_list:
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(train_X, train_Y)
        train_err = sum(svm_Y != y for svm_Y, y in zip(clf.predict(train_X), train_Y))
        train_error_rates.append((c, train_err / train_size))
        dev_err = sum(svm_Y != y for svm_Y, y in zip(clf.predict(dev_X), dev_Y))
        dev_error_rates.append((c, dev_err / dev_size))

    plot_error_rates([train_error_rates, dev_error_rates], "Error Rate", ["Training error rate", "Dev error rate"])

# 1.8
elif arg1 == '1.8':
    train_X, train_Y = zip(*train_data)
    train_size = len(train_Y)
    example_size = [5, 50, 500, 5000, 25000]
    train_times = []
    clf = svm.SVC(kernel='linear', C=1)

    for e in example_size:
        if e == 5:
            train_X_tmp = train_X[3:e + 3]
            train_Y_tmp = train_Y[3:e + 3]
        else:
            train_X_tmp = train_X[:e]
            train_Y_tmp = train_Y[:e]
        begin_time = time.time()
        clf.fit(train_X_tmp, train_Y_tmp)
        end_time = time.time()
        print end_time - begin_time
        train_times.append((e, end_time - begin_time))

    plot_error_rates([train_times], "Training Time", ["Training Time"])

