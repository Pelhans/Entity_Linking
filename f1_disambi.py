#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gen_ner import read_dataset

def disambi_f1(file):
    label_list = [0, 1]
    datasets = read_dataset(file)
    TP = FP = FN = 0 
    for idx, data in enumerate(datasets):
        pred = int(eval(data.split("\t")[0]))
        gold = int(eval(data.split("\t")[1]))
        if pred == 1 and pred == gold:
            TP += 1
        elif pred == 1 and pred != gold:
            FP += 1
        elif pred == 0 and pred != gold:
            FN += 1
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    print("Precision is: {}\nRecall is: {}\nF1 score is: {}".format(P, R, F1))
    return P, R, F1

if __name__ == "__main__":
#    disambi_f1("./models/disambi/test_results_epoch_0.tsv")
    disambi_f1("./models/disambi/test_results.tsv")
