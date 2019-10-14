#!/usr/bin/env python3
# coding=utf-8
 
""" Packaging the model and providing a unified predictive function interface  """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
import json

base_path = "/".join(os.path.realpath(__file__).split("/")[:-1])

import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
from tokenizer import Tokenizer
import tokenization
import train_ner
from train_ner import NERProcessor, word2id
from micro_f1 import decode_ner
import ner_config
from gen_ner import read_dataset
from candidate_entity_generation import gen_candidate_entity
from subject_id2text import hash_id2abstract
import subprocess
import _pickle as pkl
from build_name_dict import build_name_dict
#from run_classifier import DisambiProcessor
from disambi_processor import disambi_convert_single_example, DisambiProcessor

def preprocess_ner(query, processor, label_tag2id, wordid_map, tokenizer, task_name="ner"):
    """Preprocess data for BERT,  The first run after startup will
        be slow due to tf. contrib. util. make_tensor_proto
    :return: dict contains input_ids, input_mask, label_ids, schema_label_ids,
            segment_ids, offset, sequence_length
    :rtype: tf.contrib.util.make_tensor_proto
    """
    if not isinstance(query, str):
        raise ValueError("Input must be string")

    label = ["O"] * len(query)
    length = len(query) 
    query = ["\t".join([query, str(label)])]

    example = processor._create_examples(query)[0]
    feature, _, _ = train_ner.convert_single_example(0, example, label_tag2id, 
                                            ner_config.max_seq_length, wordid_map, tokenizer)
    inputs = trans_inputs(feature, task_name=task_name)
    return inputs, length

def preprocess_disambi(data, processor, label_list, wordid_map, tokenizer, task_name="disambi",
                  max_seq_length=200):
    """Preprocess data for BERT,  The first run after startup will
        be slow due to tf. contrib. util. make_tensor_proto
    :return: dict contains input_ids, input_mask, label_ids, schema_label_ids,
            segment_ids, offset, sequence_length
    :rtype: tf.contrib.util.make_tensor_proto
    """

    example = processor._create_examples([data])[0]
    feature = disambi_convert_single_example(0, example, label_list,  max_seq_length, tokenizer)
    inputs = trans_inputs(feature, task_name=task_name)
    return inputs

def trans_inputs(feature, task_name):
    inputs = {
        'input_ids': tf.contrib.util.make_tensor_proto([feature.input_ids],
                dtype=tf.int64),
        'input_mask': tf.contrib.util.make_tensor_proto([feature.input_mask],
                dtype=tf.int64),
        'label_ids': tf.contrib.util.make_tensor_proto([feature.label_ids],
                dtype=tf.int64),
        'segment_ids': tf.contrib.util.make_tensor_proto([feature.segment_ids],
                dtype=tf.int64),}
    if task_name == "ner":
        inputs['sequence_length'] = tf.contrib.util.make_tensor_proto([feature.sequence_length], dtype=tf.int64)
    return inputs

def gen_disambi(entity, query, offset, name_dict, id2abstract):
    candi_entity = gen_candidate_entity(entity, name_dict, mode="exact")
    datasets = []
    for centity_id in candi_entity:
        out_line = {"query_entity": entity, "query_text": query, 
                    "query_offset": offset, "candi_id": centity_id}
        out_line["candi_entity"], out_line["candi_abstract"] = id2abstract[centity_id]
        out_line["tag"] = "0"
        datasets.append(out_line)
    return datasets

def load_disambi_dict(path_id2abstract_pkl="./data/id2abstract.pkl",
                      path_name_dict="./data/name_dict.pkl",
                      path_kb_data="./original_data/kb_data",
                     ):
    if not os.path.exists(path_id2abstract_pkl):
        print("Building id2abstract.pkl...")
        start = time.time()
        id2abstract = hash_id2abstract("./original_data/kb_data", id2abstract_pkl)
        print("Build id2abstract.pkl done!,  Total time {} s".format(time.time()-start))
    else:
        id2abstract = pkl.load(open(path_id2abstract_pkl, "rb"))

    if not os.path.exists(path_name_dict):
        print(" The name dictionary does not exist and is being created. ")
        build_name_dict(path_kb_data, path_name_dict)
    name_dict = pkl.load(open(path_name_dict, "rb"))
    return name_dict, id2abstract

def predict(query, model_name, stub, disambi_stub, ner_processor, disambi_processor, bert_tokenizer,
            label_tag2id, wordid_map, label_id2tag, tokenizer,
           name_dict, id2abstract):
    """ Return the quert parsing result with the return of BERT 
    :param query: query text from api input
    :type query: string
    :param model_name: model_name name in tf_serving model_config_file
    :type model_name: string
    :param stub: prediction_service_pb2.beta_create_PredictionService_stub
    :type stub:
    :param processor: processor for query input, return InputExample class
    :type processor: class InputExample
    :param label_list: BMES label list
    :type label_list: list
    :param wordid_map: hash map from word to id
    :type wordid_map: dict
    :param label_id2tag: hash map from id to BMES label
    :type label_id2tag: dict
    """
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    inputs, length = preprocess_ner(query, ner_processor, label_tag2id, wordid_map, tokenizer)
    for k, v in inputs.items():
        request.inputs[k].CopyFrom(v)
    start = time.time()
    result = stub.Predict(request, 60.0).outputs
    pred_ids = result["pred_ids"].int_val[1:length+1]
    pred_ids = [str(p) for p in pred_ids]
    ner_offset = decode_ner(pred_ids)
    print(" full_query: {}\n ner_tag: {}".format(query, ner_offset) )
    print("Predict time: {} s".format(time.time()-start))
    
    disambi_label_list = ["0", "1"]
    output = {"query": query,
              "mention_data":[  ]}
    for start, end in ner_offset:
        start, end = int(start), int(end)
        entity = query[start: end+1]
        disambi_datasets = gen_disambi(entity, query, [start, end], name_dict, id2abstract)
        disambi_result = []
        for disambi_data in disambi_datasets:
            inputs = preprocess_disambi(disambi_data, disambi_processor, disambi_label_list, wordid_map, bert_tokenizer)
            request = predict_pb2.PredictRequest()
            request.model_spec.name = "poi_disambi"
            for k, v in inputs.items():
                request.inputs[k].CopyFrom(v)
            result = disambi_stub.Predict(request, 60.0).outputs
            pred_ids = result["probabilities"].float_val
            disambi_result.append(pred_ids[1])
        found_candi = (np.asarray(disambi_result) > 0.5).sum() > 0
        candi_entity_idx = np.argmax(disambi_result)
        candi_entity = disambi_datasets[candi_entity_idx]["candi_entity"]
        confidence =  disambi_result[candi_entity_idx]
        if found_candi:
            candi_entity_abstract = disambi_datasets[candi_entity_idx]["candi_abstract"]
            candi_entity_id = disambi_datasets[candi_entity_idx]["candi_id"]
        else:
            candi_entity_abstract = ""
            candi_entity_id = "NIL"
            confidence = 1 - confidence

        output["mention_data"].append({"mention_data": candi_entity,
                                     "offset": start,
                                     "kb_id": candi_entity_id,
                                     "object": candi_entity_abstract,
                                     "confidence": confidence})
    return output

class Client:
    """Predict model Client
    :param model_name:  Model name set in tf_service 
    :type model_name: string
    :param server_ip: tf_service's ip
    :type server_ip: string
    :param server_port: tf_service's port
    :type server_port: string
    :param es_server: ES's ip:port
    """
    def __init__(self, model_name="poi_ner", server_ip="192.168.31.187", server_port=9209,
                disambi_model_name="poi_disambi", disambi_server_port=9210): 
        self.model_name = model_name
        self.disambi_model_name=disambi_model_name
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.disambi_server_port = int(disambi_server_port)

        self.channel = implementations.insecure_channel(self.server_ip, self.server_port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        
        self.disambi_channel = implementations.insecure_channel(self.server_ip, self.disambi_server_port)
        self.disambi_stub = prediction_service_pb2.beta_create_PredictionService_stub(self.disambi_channel)

        self.ner_processor = NERProcessor()
        self.disambi_processor = DisambiProcessor()
        self.label_list = self.ner_processor.get_labels()
        self.label_id2tag = {i: v for i,v in enumerate(self.label_list)}
        self.label_tag2id = {v: i for i,v in enumerate(self.label_list)}
        self.wordid_map = word2id("./pb_model/char_vocab.txt")
        self.tokenizer = Tokenizer(self.wordid_map)
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file="./pb_model/char_vocab.txt", do_lower_case=True)
        self.name_dict, self.id2abstract = load_disambi_dict()

    def query_parsing(self, query):
        """ Functional interface provided to the outside
        :param query: query poi from get method
        :type query: string
        :param city:  Query poi's city
        :type city: string
        :return: a dict with lon, lat and poi type
        :rtype: dict
        """
        if not query:
            raise ValueError("Query text is empty !!")

        return predict(query, self.model_name, self.stub, self.disambi_stub, self.ner_processor,
                       self.disambi_processor, self.bert_tokenizer,
                       self.label_tag2id, self.wordid_map, self.label_id2tag,
                      self.tokenizer, self.name_dict, self.id2abstract)

if __name__ == "__main__":
    client = Client(model_name="poi_ner",
                    server_ip="192.168.31.187",
                    server_port=9209,
                    disambi_model_name="poi_disambi", 
                    disambi_server_port=9210)

    res = client.query_parsing("游戏《英雄联盟》胜利系列限定皮肤")

    print("res: ", res)
