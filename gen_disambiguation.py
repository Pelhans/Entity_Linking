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
from tqdm import tqdm
from gen_ner import read_dataset
from candidate_entity_generation import gen_candidate_entity
from subject_id2text import hash_id2abstract

def gen_disambi(infile, outfile):
    id2abstract_pkl = "./data/id2abstract.pkl"

    if not os.path.exists("./data/disambi/"):
        subprocess.getoutput("mkdir ./data/disambi/")
    datasets = read_dataset(infile)
    if not os.path.exists(id2abstract_pkl):
        print("Building id2abstract.pkl...")
        start = time.time()
        id2abstract = hash_id2abstract("./original_data/kb_data", id2abstract_pkl)
        print("Build id2abstract.pkl done!,  Total time {} s".format(time.time()-start))
    outwriter = open(outfile, "w")
    for data in tqdm(datasets):
        data = eval(data)
        candi_text = data["text"]
        for mention in mention_data:
            if mention["kb_id"] == "NIL":
                continue
            source_entity = mention["mention"]
            offset = int(mention["offset"])
            candi_offset = (offset,  len(source_entity) + offset)
            candi_entity = gen_candidate_entity(source_entity, mode="exact_match")
            if not candi_entity:
                continue
            if len(candi_entity) > 20:
                print("Long candi_entity: ", len(candi_entity))
            for centity_id in candi_entity:
                out_line = {"candi_entity": candi_entity, "candi_text": candi_text,
                            "candi_offset": candi_offset}
                out_line["subject"], out_line["abstract"] = id2abstract(centity_id)
                out_line["tag"] = 1 if centity_id == mention["kb_id"] else 0
                outwriter.write(json.dumps(out_line))

if __name__ == "__main__" :
    gen_disambi("./original_data/train_pre.json", 
               "./data/disambi/all.txt", )
