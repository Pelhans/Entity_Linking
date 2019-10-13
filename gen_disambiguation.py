#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import json
import time
import codecs
import logging
import _pickle as pkl
from tqdm import tqdm
from gen_ner import read_dataset
from candidate_entity_generation import gen_candidate_entity
from subject_id2text import hash_id2abstract

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name_dict_path", type=str, default="./data/name_dict.pkl")
parser.add_argument("--kb_data_path", type=str, default="./original_data/kb_data")
args = parser.parse_args()

def gen_disambi(infile, outfile):
    id2abstract_pkl = "./data/id2abstract.pkl"

    if not os.path.exists("./data/disambi/"):
        subprocess.getoutput("mkdir ./data/disambi/")
    datasets = read_dataset(infile)
    if not os.path.exists(id2abstract_pkl):
        logging.info("Building id2abstract.pkl...")
        start = time.time()
        id2abstract = hash_id2abstract("./original_data/kb_data", id2abstract_pkl)
        logging.info("Build id2abstract.pkl done!,  Total time {} s".format(time.time()-start))
    else:
        id2abstract = pkl.load(codecs.open("./data/id2abstract.pkl", "rb"))

    if not os.path.exists(args.name_dict_path):
        logging.info(" The name dictionary does not exist and is being created. ")
        build_name_dict(args.kb_data_path, args.name_dict_path)
    name_dict = pkl.load(open(args.name_dict_path, "rb"))

    outwriter = codecs.open(outfile, "w", "utf-8")
    pos_count = 0
    neg_count = 0
    total_entity = 0
    used_lines = 0
    max_leng = 0
    for data in tqdm(datasets):
        data = eval(data)
        candi_text = data["text"]
        for mention in data["mention_data"]:
            if mention["kb_id"] == "NIL":
                continue
            source_entity = mention["mention"]
            offset = int(mention["offset"])
            candi_offset = (offset,  len(source_entity) + offset)
            candi_entity = gen_candidate_entity(source_entity, name_dict, mode="exact")
            used_lines += 1
            total_entity += len(candi_entity)
            if not candi_entity:
                continue
            if len(candi_entity) > 20:
                max_leng += 1
#                continue
            for centity_id in candi_entity:
                if centity_id not in id2abstract:
                    continue
                out_line = {"query_entity": source_entity, "query_text": candi_text,
                            "query_offset": candi_offset}
                out_line["candi_entity"], out_line["candi_abstract"] = id2abstract[centity_id]
                if centity_id == mention["kb_id"]:
                    out_line["tag"] = 1
                    pos_count += 1
                else:
                    out_line["tag"] = 0
                    neg_count += 1
#                out_line["tag"] = 1 if centity_id == mention["kb_id"] else 0
                outwriter.write(json.dumps(out_line) + "\n")
    logging.info("upper max_length: {}".format(max_leng))
    logging.info("Communist sample {}, of which positive {}, negative {} ".format(pos_count + neg_count, pos_count, neg_count))
    logging.info("Avg candidate entity length: {}".format(total_entity/used_lines))

def divide_set(infile):
    logging.info("Dividing file into train/dev/test...")
    train_writer = codecs.open("./data/disambi/train.txt", "w", "utf-8")
    dev_writer = codecs.open("./data/disambi/dev.txt", "w", "utf-8")
    test_writer = codecs.open("./data/disambi/test.txt", "w", "utf-8")
    datasets = read_dataset(infile)
    total_line = int(subprocess.getoutput("wc -l {}".format(infile)).split()[0])
#    total_line = 300000
    logging.info("total_line: {}".format(total_line))
    for idx, data in enumerate(datasets):
        if idx > total_line:
            break
        if idx < 0.8 * total_line:
            train_writer.write(data)
        elif idx < 0.9 * total_line:
            dev_writer.write(data)
        elif idx < total_line:
            test_writer.write(data)
    logging.info("Done")

if __name__ == "__main__" :
    logging.basicConfig(level = logging.DEBUG,
                       format = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    gen_disambi("./original_data/train_pre.json",  "./data/disambi/all.txt", )
    divide_set("./data/disambi/all.txt")
