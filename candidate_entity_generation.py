#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import _pickle as pkl
import time
from build_name_dict import build_name_dict
from cal_similarity import dice, edit_distance

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name_dict_path", type=str, default="./data/name_dict.pkl")
parser.add_argument("--kb_data_path", type=str, default="./original_data/kb_data")
args = parser.parse_args()


def gen_candidate_entity(entity, mode, dice_value=0.8, edit_value=1):
    """ Candidate entity generation 
    :param mode: one of "exact_match"/ "exact_dice"/ "exact_dice_edit"
        if mode == "exact_dice", you must set the value of dice
        if mode == "exact_dice_edit", you must set the value of dice and edit
    """
    if not isinstance(entity, str):
        entity = str(entity)
    if not os.path.exists(args.name_dict_path):
        print(" The name dictionary does not exist and is being created. ")
        build_name_dict(args.kb_data_path, args.name_dict_path)

    name_dict = pkl.load(open(args.name_dict_path, "rb"))
    # !!  Traveling through all entities ----> low efficient 
    #  Exact Matching + Partial Matching 
    #  Partial matching needs to be satisfied: 
    #    1) Editing distance is less than 2 or 
    #    2) Dice coefficient is greater than 0.8 
    candidate_id = []
    for  cand_entity in name_dict:
        if mode == "exact_match" and entity == cand_entity:
            candidate_id.extend(name_dict[cand_entity])
        elif mode == "exact_dice" and dice(entity, cand_entity) >= dice_value:
            candidate_id.extend(name_dict[cand_entity])
        elif mode == "exact_dice_edit" and dice(entity, cand_entity) >= dice_value or \
                edit_distance(entity, cand_entity) <= edit_value:
            candidate_id.extend(name_dict[cand_entity])
    return candidate_id

if __name__ == "__main__":
    start = time.time()
    print(" Candidate entity ID: ", gen_candidate_entity("香港周永康", mode="exact_match"))
    print("time: ", time.time()-start, " s")
