# 2019 中文短文本的实体链指

* NER 部分是随手写的一个 baseline, F1 为 ~0.83, 模型目前是 BERT + CRF.  除此之外还实现了 BERT+BLSTM+CRF.    
    * 标签为 BIO， 训练数据是根据 train.json， 将其中被标记的实体位置打上 BI 标签    
* 实体消岐部分采用 BERT+Dense 的 二分类模型, F1 为 0.896
    * 候选实体根据精确匹配生成(candidate_entity_generation.py 里也实现了 Dice 系数, Jaccard 系数, 编辑距离筛选)    
    * 平均候选实体个数为 5.7 个， 最终训练集大小为 116 万左右    
    * 模型输入为 textA = mention 的句子, textB = 候选实体所有 属性-值组成的句子

# 运行

## NER 部分

* 下载 百度官方数据集, 解压到 original_data 目录下    
* 下载 bert 模型放到 hit_bert/ 目录下    
* 运行 python3 gen_ner.py 生成训练数据    
* 运行 python3 train_ner.py 训练数据

## 实体消岐部分

* 运行 gen_disambiguation.py 产生消岐数据集    
* 修改 run.sh 中的参数和环境变量    
* 运行 ./run.sh 进行训练    
* 运行 f1_disambi.py 计算测试结果的 F1 值
