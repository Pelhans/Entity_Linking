#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess as sp
import codecs
import os
import json
import tokenization
from tokenization import BasicTokenizer
from clean import punctuation_chinese2english

def read_dataset(file):
    if not os.path.exists(file):
        raise ValueError("File not exist")

    total_line = int(sp.getoutput("wc -l {}".format(file)).split()[0])
    with codecs.open(file, "r", "utf-8") as inf:
        for idx in range(total_line):
            yield inf.readline().strip()

def get_labels(task_name):
    if task_name.lower() == "ner":
        return ["B", "I", "O"]
    else:
        raise ValueError("Not implement error")

def tag_ner(infile, outfile):
    tokenizer = BasicTokenizer()
    datasets = read_dataset(infile)
    label_list = get_labels(task_name="ner")
    writer = codecs.open(outfile, "w", "utf-8")
    for idx, data in enumerate(datasets):
        try:
            data = json.loads(data)
        except:
            print("current idx: {}. text is: {}".format(idx, data))
            continue
        text = tokenization.convert_to_unicode(data["text"])
        new_text = [punctuation_chinese2english(t) for t in text]
        new_text = "".join(new_text)
        label = ["O"]*len(list(new_text))
        for mention in data["mention_data"]:
            offset_start = int(mention["offset"])
            offset_end = offset_start + len(mention["mention"])
            label[offset_start] = "B"
            label[offset_start+1:offset_end] = ["I"]*(len(mention["mention"])-1)
        # 防止开头有特殊符号的情况
        if new_text[0] == " ":
            label.pop(0)
        writer.write(new_text + "\t" + str(label) + "\n")

        if idx <= 5:
            print("输入文本: {},\n对应标签: {}".format(text, str(label)))
    writer.close()

def divide(path, infile):
    total_line = int(sp.getoutput("wc -l {}".format(infile)).split()[0])
    datasets = read_dataset(infile)
    
    train_writer = codecs.open(os.path.join(path, "train.csv"), "w", "utf-8")
    dev_writer = codecs.open(os.path.join(path, "dev.csv"), "w", "utf-8")
    test_writer = codecs.open(os.path.join(path, "test.csv"), "w", "utf-8")
    for idx, data in enumerate(datasets):
        if idx < 0.8*total_line:
            train_writer.write(data + "\n")
        elif idx < 0.9*total_line:
            dev_writer.write(data + "\n")
        else:
            test_writer.write(data + "\n")
    train_writer.close()
    dev_writer.close()
    test_writer.close()

if __name__ == "__main__":
    tag_ner("./original_data/train.json", "./data/ner/tag_ner.txt")
    divide("./data/ner/",  "./data/ner/tag_ner.txt")

