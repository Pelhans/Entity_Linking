#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gen_ner import read_dataset
import _pickle as pkl

def hash_id2abstract(kb_data, id2abstract_pkl):
    datasets = read_dataset(kb_data)
    id2abstract = {}
    for data in datasets:
        data = eval(data)
        if len(data["data"]) == 0:
            continue
        if data["data"][0]["predicate"] == "摘要":
            id2abstract[data["subject_id"]] = (data["subject"], data["data"][0]["object"])
    pkl.dump(id2abstract, open(id2abstract_pkl, "wb"))

