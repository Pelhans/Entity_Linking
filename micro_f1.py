#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gen_ner import read_dataset, get_labels

def micro_f1(file):
    label_list = get_labels(task_name="ner")
    id2label = {i: v for i,v in enumerate(label_list)}
    datasets = read_dataset(file)
    TP = FP = FN = 0 
    for idx, data in enumerate(datasets):
        TP, FP, FN = cal_tp_fp_fn(idx, data,TP, FP, FN)
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    print("Precision is: {}\nRecall is: {}\nF1 score is: {}".format(P, R, F1))
    return P, R, F1

def cal_tp_fp_fn(idx, line, TP, FP, FN):
    predict = eval(line.split("###")[0])
    gold = eval(line.split("###")[1])
    
    decode_pred = decode_ner(predict)
    decode_gold = decode_ner(gold)
    if idx < 5:
        print("predict: ", predict)
        print("decode_pred: ", decode_pred)
        print("gold: ", gold)
        print("decode_gold: ", decode_gold)

    for idx,ptag in enumerate(decode_pred):
        if ptag in decode_gold:
            TP += 1
        else:
            FP += 1
    for idx,gtag in enumerate(decode_gold):
        if gtag not in decode_pred:
            FN += 1
    return TP, FP, FN

    
def decode_ner(line):
    res = []
    idx = 0
    while idx < len(line):
        if line[idx] == "1":
            start = idx
            idx += 1
            while idx < len(line) and line[idx] == "2":
                idx += 1
            res.append([start, idx-1])
        idx += 1
    return res

if __name__ == "__main__":
    micro_f1("./models/test_results.tsv")
