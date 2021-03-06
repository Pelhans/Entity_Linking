#!/usr/bin/env python3
# coding=utf-8

"""Model dec """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from ner_config import *
import modeling
from modeling import layer_norm_and_dropout, layer_norm
import numpy as np

def bert_blstm_crf(bert_config, is_training, input_ids, segment_ids,input_mask,
               label_ids, sequence_length, num_labels,  use_one_hot_embeddings):
    """combine bert + blstm + crf_layer

    :param bert_config: bert_config from model config file
    :type bert_config: dict
    :param is_training: train state
    :type is_training: bool
    :param input_ids: input text ids for each char
    :type input_ids: list
    :param segment_ids: 0 for first sentence and 1 for second sentence,
                        for this task, all is 0, length is max_seq_length
    :type segment_ids: list
    :param input_mask: mask for sentence to suit bert model,
                        for this task, all is 1, length is max_seq_length
    :type input_mask: list
    :param label_ids: BIO labels ids
    :type label_ids: list
    :param sequence_length: sequence length for each input sentence before padding
    :type sequence_length: list, [lengh_sentence1, 2,..]
    :param num_labels: nums of BIO labels
    :type num_labels: int
    :param use_one_hot_embeddings: wehter use_one_hot_embeddings
    :type use_one_hot_embeddings: bool
    :return: total_loss, per_example_loss, logits for ner, pred_ids using viterbi
    :rtype: tuple
    """

    batch_size = tf.shape(input_ids)[0]
    bert_out = bert(bert_config, is_training, input_ids,
                    input_mask, segment_ids, use_one_hot_embeddings)
    y_pred = blstm(is_training, bert_out)
    if is_training:
        y_pred = layer_norm_and_dropout(y_pred, 0.5)
    else:
        bert_out = layer_norm(bert_out)
    hidden_size = tf.shape(y_pred)[-1]
    blstm_out = linear_layer(y_pred, hidden_size, num_labels, "linear")
    crf_out = crf_layer(blstm_out, label_ids, batch_size, 
                          sequence_length, num_labels, max_seq_length, "crf")
    return crf_out


def bert_crf(bert_config, is_training, input_ids, segment_ids,input_mask,
               label_ids, sequence_length, num_labels,  use_one_hot_embeddings):

    batch_size = tf.shape(input_ids)[0]
    bert_out = bert(bert_config, is_training, input_ids,
                    input_mask, segment_ids, use_one_hot_embeddings)
#    hidden_size = tf.shape(bert_out)[-1]
    hidden_size = 768
    if is_training:
        bert_out = layer_norm_and_dropout(bert_out, 0.5)
    else:
        bert_out = layer_norm(bert_out)
    bert_out = tf.reshape(bert_out, [-1, hidden_size])
    linear_out = linear_layer(bert_out, hidden_size, num_labels, "linear")
    crf_out = crf_layer(linear_out, label_ids, batch_size, 
                          sequence_length, num_labels, max_seq_length, "crf")
    return crf_out

def bert_mlp(bert_config, is_training, input_ids, segment_ids,input_mask,
               label_ids, sequence_length, num_labels,  use_one_hot_embeddings):

    bert_out = bert(bert_config, is_training, input_ids,
                    input_mask, segment_ids, use_one_hot_embeddings, task_type="binary")
#    hidden_size = tf.shape(bert_out)[-1]
    hidden_size = 768
    bert_out = tf.reshape(bert_out, [-1, hidden_size])
    if is_training:
         bert_out = tf.nn.dropout(bert_out, keep_prob=0.9)
    logits = linear_layer(bert_out, hidden_size, num_labels, "linear")
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probabilities = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
#        probs = tf.nn.softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
#        loss = focal_loss(probs, one_hot_labels)
#        loss = multi_category_focal_loss2_fixed(probs, one_hot_labels)
#    return (loss, loss, logits, predictions)
    return (loss, loss, logits, probabilities)

def focal_loss(y_pred, y_gold, gamma=2, alpha=0.25):
    return -tf.reduce_mean(alpha * tf.multiply(tf.pow(y_gold, gamma), tf.log(y_pred)))

def multi_category_focal_loss2_fixed(y_pred, y_true):
    epsilon = 1.e-7
    gamma=2.
    alpha = tf.constant(0.5, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl)
    return loss

def weight_variable(shape):
#    initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.get_variable("weight",
                              shape, 
                              initializer=tf.truncated_normal_initializer(stddev=0.02))
    return initial

def bias_variable(shape):
    initial = tf.get_variable("bias",
                              shape,
                              initializer=tf.zeros_initializer())
#    initial = tf.constant(0.1, shape=shape)
    return initial

def lstm_cell(hidden_size):
    cell = rnn.LSTMCell(hidden_size,
                        reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=True)

def bert(bert_config, is_training, input_ids, input_mask,
         segment_ids, use_one_hot_embeddings, task_type="sequence"):
    """Use bert model to get sequence output"""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,)

    # get sequence output
    # output_layer's shape [batch_size, sequence_length, hidden_size]
    # where hidden_size is 768
    if task_type == "sequence":
        output_layer = model.get_sequence_output()
    elif task_type == "binary":
        output_layer = model.get_pooled_output()
    return output_layer

def blstm(is_training, inputs):
    """BLSTM model"""

    batch_size = tf.shape(inputs)[0]
    hidden_size = hidden_size_blstm

    cell_fw = rnn.MultiRNNCell([lstm_cell(hidden_size) for _ in range(layer_num)],
                               state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell(hidden_size) for _ in range(layer_num)],
                               state_is_tuple=True)

    # shape c_state and h_state [batch_size, hidden_size],
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    with tf.variable_scope('bidirectional_rnn'):
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(max_seq_length):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                # output_fw -> h_{t}, state_fw -> cell_{t}
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(max_seq_length):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        outputs_bw = tf.reverse(outputs_bw, [0])
        # after concat, shape of output [timestep, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, [-1, hidden_size*2])

        return output

def linear_layer(input, input_size, num_labels, name):
    with tf.variable_scope(name):
        softmax_w = weight_variable([input_size, num_labels])
        softmax_b = bias_variable([num_labels])
        output = tf.matmul(input, softmax_w) + softmax_b
        return output

def crf_layer(y_pred, label_ids, batch_size, sequence_length, num_labels, max_seq_length, name):
    with tf.variable_scope(name):
        sequence_length = tf.squeeze(sequence_length)
    
        logits = tf.reshape(y_pred, [-1, max_seq_length, num_labels])
        loss, trans = _crf_layer(logits, num_labels, label_ids,
                                length=sequence_length, name="crf_bio")
        
        sequence_length = tf.reshape(sequence_length, [batch_size])
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans,
                                     sequence_length=sequence_length)
    
        return loss, loss, logits, pred_ids

def _crf_layer(logits, num_labels, label_ids, length, name):
    """Calculate the likelihood loss function with CRF layer"""
    with tf.variable_scope(name):
        log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=label_ids,
            sequence_lengths=length)
        return tf.reduce_mean(-log_likelihood), trans
