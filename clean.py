#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata

def punctuation_chinese2english(char):
    """trans Chinese punctuation to English punctuation"""
    punc_map = {"！": "!",
                '“': '"',
                "”": '"',
                "￥": "$", 
                "’": "'", 
                "‘": "'", 
                "（": "(", 
                "）": ")", 
                "，": ",", 
                "。": ".", 
                "、": "/", 
                "：": ":",
                "；": ";",
                "？": "?",
                "《": "<",
                "》": ">",
                "【": "[",
                "】": "]",
                "…": "...",
                "——": "-",
                "·": " ",
                "｛": "{",
                "｝": "}",
                "〈": "<",
                "〉": ">",
                "﹞": ")",
                "﹏": "_",
                "「": "[",
                "〞": '"',
                "﹚": ")",
                "„": ",",
                "〜": " ",
                "﹜": "}",
                "〔": "(",
                "」": "]",
                "〝": '"',
                "（": "(",
                "〗": "]",
                "﹣": "-",
                "﹐": ",",
                "﹒": " ",
                "※": " ",
                "‼": "!",
                "”": '"',
                "﹪": "%",
                "〖": "[",
                "﹙": "(",
                "？": "?",
                "）": ")",
                "﹕": ":",
                "»": ">",
                "†": "+",
                "﹝": "(",
                "﹖": "?",
                "‖": "|",
                "〕": ")",
                "﹠": "&",
                "『": "[",
                "︱": "|",
                "‹": "<",
                "』": "]",
                "〈": "<",
                "•": " ",
                "‾": "-",
                "«": "<",
               }
    if char in punc_map:
        return punc_map[char]
    # 是标点符号， 但未被收录进 map， 返回空格
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return " "
    else:
        # 不是标点符号就返回原字符
        return char
