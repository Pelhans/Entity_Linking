# 2019 中文短文本的实体链指

目前随手写了一个NER 部分的 baseline, F1 为 0.82, 模型目前是 BERT + CRF.  除此之外还实现了 BERT+BLSTM+CRF.

# 运行

## NER 部分

* 下载 百度官方数据集, 解压到 original_data 目录下    
* 下载 bert 模型放到 hit_bert/ 目录下    
* 运行 python3 gen_ner.py 生成训练数据    
* 运行 python3 train_ner.py 训练数据
