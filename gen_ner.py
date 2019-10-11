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
import _pickle as pkl
import logging
import tokenization
from tokenization import BasicTokenizer
from clean import punctuation_chinese2english

def read_dataset(file):
    if not os.path.exists(file):
        raise ValueError("File not exist")

    total_line = int(sp.getoutput("wc -l {}".format(file)).split()[0])
    with codecs.open(file, "r", "utf-8") as inf:
        for idx in range(total_line):
            yield inf.readline()

def get_labels(task_name):
    if task_name.lower() == "ner":
        return ["B", "I", "O"]
    else:
        raise ValueError("Not implement error")

def standard_string(string, lower=True):
    string = string.lower() if lower else string
    string = tokenization.convert_to_unicode(string)
    string = [punctuation_chinese2english(t) for t in string]
    return "".join(string)

def build_kb(kb_file, outfile):
    """ Preprocess kb_data and store it as pkl file """
    pass

def tag_ner(infile, outfile, name_dict_file):
#    tokenizer = BasicTokenizer()
    datasets = read_dataset(infile)
#    label_list = get_labels(task_name="ner")
    name_dict = pkl.load(open(name_dict_file, "rb"))
    writer = codecs.open(outfile, "w", "utf-8")
    merge_count = 0
    for idx, data in enumerate(datasets):
        try:
            data = json.loads(data)
        except:
            print("current idx: {}. text is: {}".format(idx, data))
            continue
#        new_text = standard_string(data["text"])
        new_text = data["text"]
        label = ["O"]*len(new_text)
        for mention in data["mention_data"]:
            offset_start = int(mention["offset"])
            offset_end = offset_start + len(mention["mention"])
            label[offset_start] = "B"
            label[offset_start+1:offset_end] = ["I"]*(len(mention["mention"])-1)
        assert len(new_text) == len(label)
        # 融合实体, 统一例如 "高清视频" 被标记成 "高清" 和 “视频” 两个实体的情况
#        label, merge_count =  merge_entity(new_text, label, name_dict, merge_count)
        writer.write(new_text + "\t" + str(label) + "\n")
        if idx <= 5:
            print("输入文本: {},\n对应标签: {}".format(new_text, str(label)))
    writer.close()

def merge_entity(text, label, name_dict, merge_count):
    if isinstance(text, list):
        text = [str(t) for t in text]
        text = "".join(text)
    try: 
        assert len(text) == len(label)
    except:
        text = text[1:]
    entity_start = 0
    entity_end = 0
    merge_offset = []
    id = -1
    while id < len(label)-1:
        id += 1
        if label[id] == "B":
            if entity_start != entity_end:
                merge_count += 1
                entity_end = id
            else:
                entity_start = entity_end = id
        elif label[id] == "O":
            if entity_end != entity_start:
                merge_offset.append([entity_start, entity_end])
            entity_start = entity_end = id
        elif id == len(label) - 1:
            merge_offset.append([entity_start, id])
        elif label[id] == "I":
            entity_end = id
    logging.debug("merge_offset: ", merge_offset)
    for entity_start, entity_end in merge_offset:
        front_start = entity_start
        while front_start < entity_end+1:
            entity = text[front_start: entity_end+1]
            _, tmp_end = front_max_match_once(entity, name_dict)
            label[front_start: front_start + tmp_end+1] = ["B"] + ["I"]*(tmp_end)
            front_start = front_start + tmp_end + 1
            entity_start = front_start
    return label, merge_count

def front_max_match(text, name_dict):
    start = 0
    end = len(text)
    offset = []
    while start != len(text):
        if text[start: end] in name_dict:
            offset.append([start, end-1])
            start = end
            end = len(text)
        elif start == end:
            offset.append([start, end])
            start += 1
            end = len(text)
        else:
            end -= 1
    return offset

def front_max_match_once(text, name_dict):
    start = 0
    end = len(text)
    while start != end-1:
        if text[start: end] in name_dict:
            return [start, end-1]
        else:
            end -= 1
    # if not found
    return [start, len(text)-1]

def divide(path, infile):
    total_line = int(sp.getoutput("wc -l {}".format(infile)).split()[0])
    datasets = read_dataset(infile)
    
    train_writer = codecs.open(os.path.join(path, "train.csv"), "w", "utf-8")
    dev_writer = codecs.open(os.path.join(path, "dev.csv"), "w", "utf-8")
    test_writer = codecs.open(os.path.join(path, "test.csv"), "w", "utf-8")
    for idx, data in enumerate(datasets):
        if idx < 0.8*total_line:
            train_writer.write(data)
        elif idx < 0.9*total_line:
            dev_writer.write(data)
        else:
            test_writer.write(data)
    train_writer.close()
    dev_writer.close()
    test_writer.close()

if __name__ == "__main__":
    name_dict_file = "./data/name_dict.pkl"
    ori_train_file = "./original_data/train.json"
    outfile = "./data/ner/tag_ner.txt"
    tag_ner(ori_train_file, outfile, name_dict_file)
    divide("./data/ner/",  "./data/ner/tag_ner.txt")

    # 测试无融合实体
#    text = "《今日说法》 20140121 飞贼撞天网_今日说法"
#    label = ["O", "B", "I", "I", "I", "O", "O", "O", "O", "O","O","O","O","O","O","O", "B", "I", "O", "B", "I", "O", "B", "I", "I", "I"]
    # 测试有 NIL 实体时, lg g5 没有
    text = "lg g5 评测, 玩火的阿斗,"
    label = ['B', 'I', 'I', 'I', 'I', 'O', 'B', 'I', 'O', 'B', 'I', 'I', 'I', 'I']
    # 测试融合实体情况
#    text = "高清视频在线下载"
    # 测试正向最大匹配
#    text = "在线下载高清视频"
#    label = ["B", "I", "B", "I", "B", "I", "B", "I"]
    name_dict = pkl.load(open(name_dict_file, "rb"))
    offset = front_max_match(text, name_dict)
    out = [text[s:e+1] for s,e in offset]
#    print("offset: ", "/".join(out))
    merge_count = 0
    new_label, merge_count = merge_entity(text, label, name_dict, merge_count)
#    print(" new_text: {}\n new_label: {}\n merge_count：{}".format(text, new_label, merge_count))
