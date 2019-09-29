#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gen_ner import read_dataset
from collections import defaultdict
import _pickle as pkl
from tqdm import tqdm

def build_name_dict(kb_file, name_pkl):
    name_dict = defaultdict(list)
    datas = read_dataset(kb_file)
    for data in tqdm(datas):
        data = eval(data)
        subject = data["subject"]
        subject_id = data["subject_id"]
        name_dict[subject].append(subject_id)
        for alias in data["alias"]:
            if alias != subject:
                name_dict[alias].append(subject_id)

    pkl.dump(name_dict, open(name_pkl, "wb"))

if __name__ == "__main__":
    build_name_dict(kb_file = "./original_data/kb_data", name_pkl = "./data/name_dict.pkl")
