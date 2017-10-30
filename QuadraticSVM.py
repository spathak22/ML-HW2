# !/usr/bin/env python

from __future__ import division

import sys
import numpy as np
import time
from collections import defaultdict
from sklearn import svm
import matplotlib.pyplot as plt
import csv


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

    plt.ylabel('Error Rate')
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

def print_eval_result(clf,result_list, type='Linear'):
    print(' -- Printing results for --- :',type)
    train_err = sum(svm_Y != y for svm_Y, y in zip(clf.predict(train_X), train_Y))
    print "Q1.1 train_err :: {:.2%} ".format(train_err / train_size)
    result_list +=[round((train_err/train_size)*100,5)]

    dev_err = sum(svm_Y != y for svm_Y, y in zip(clf.predict(dev_X), dev_Y))
    print "Q1.1 dev_err :: {:.2%} ".format(dev_err / dev_size)

    result_list +=[round((dev_err/dev_size)*100,5)]


    return result_list



def write_to_csv(res_list, file_name = "Result_Comp_LI_Qu.csv", sep =','):
    folder_path ="./output_data"
    mid_cha = '/'
    file_path = folder_path + mid_cha + file_name
    with open(file_path,"wb") as fp:
        csv_writer = csv.writer(fp, delimiter=sep)
        csv_writer.writerows(res_list)





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
    if arg1 == '3.1':
        train_X, train_Y = zip(*train_data)
        dev_X, dev_Y = zip(*dev_data)
        train_size = len(train_Y)
        dev_size = len(dev_Y)

        # temp_c = range(5,100,5)
        # c_list = [0.01, 0.1, 1] + temp_c

        c_list = [50]

        res_list = [['Type','C','Train_Error','Dev_Error','Training_Time','Support Vectors']]
        for c in c_list:
            print ' c = ',c
            begin_time = time.time()
            temp_list = []
            temp_list+=["Quadratic", c]
            clf1 = svm.SVC(kernel='poly', degree=2, coef0=1, C=c)

            clf1.fit(train_X, train_Y)
            end_time = time.time()
            temp_list = print_eval_result(clf1,result_list = temp_list, type="Quadratic")
            temp_list+=[end_time-begin_time]
            supportVectors = clf1.n_support_
            supportSum = sum(supportVectors)
            temp_list += [supportSum]
            print "Q1.1 Training time is :: ", end_time - begin_time
            print "Support vectors is ::  ",supportSum
            res_list+=[temp_list]
        print res_list
        #write_to_csv(res_list,file_name = "Result_Comp_LI_Qu.csv",sep=',')



