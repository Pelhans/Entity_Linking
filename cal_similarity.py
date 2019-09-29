#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import difflib

def dice(strA, strB):
    """ Cal Dice coefficient """
    overlap = intersection(strA, strB)
    return 2.0*overlap/(len(strA) + len(strB))

def Jaccard(strA, strB):
    """ Cal Jaccard coefficient """
    overlap = intersection(strA, strB)
    return overlap/(len(strA) + len(strB) - overlap)

def edit_distance(strA, strB):
    """ Cal Levenshtein distance """
    leven_cost = 0
    s = difflib.SequenceMatcher(None, strA, strB)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost

def intersection(strA, strB):
    """  Calculate the number of same strings between two strings  """
    if not strA or not strB:
        return 0
    strA_list = list(strA)
    strB_list = list(strB)
    overlap = 0
    for sa in strA_list:
        if sa in strB_list:
            overlap += 1
            strB_list.pop(strB_list.index(sa))
    return overlap


if __name__ == "__main__":
    print("Dice coefficient: ", dice("Lvensshtain", "Levenshtein"))
    print("Jaccard coefficient: ", Jaccard("Lvensshtain", "Levenshtein"))
    print("Levenshtein distance: ", edit_distance("Lvensshtain", "Levenshtein"))
