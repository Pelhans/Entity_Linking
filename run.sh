#!/bin/bash

#export BERT_BASE_DIR=/home1/peng/project/bert_poi/chinese_L-12_H-768_A-12 #全局变量 下载的预训练bert地址
export CUR_PATH=`pwd`
export BERT_BASE_DIR=$CUR_PATH/hit_bert #全局变量 下载的预训练bert地址
export MY_DATASET=$CUR_PATH/data/disambi #全局变量 数据集所在地址

#CUDA_VISIBLE_DEVICES=0

python3 run_classifier.py \
  --task_name=el \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=200 \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$CUR_PATH/models/disambi \
  --serving_model_save_path=$CUR_PATH/pb_model/
