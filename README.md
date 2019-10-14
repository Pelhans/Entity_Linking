# CCKS 2019 中文短文本的实体链指

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

# 通过 tf_serving 调用

* 下载 NER 和 实体消岐的 pb 模型, 放到 pb_model 目录    
* 确认安装 tensorflow_model_server 或者 通过 docker 使用 pb 文件开启服务    
    * NER 的端口为 9209, 模型名字为 poi_ner    
    * 消岐的端口为 9210, 模型名字为 poi_disambi    
* 运行 python3 predict.py 进行测试

例如， 输入为 "游戏《英雄联盟》胜利系列限定皮肤"

输出为: 

{ 'query': '游戏《英雄联盟》胜利系列限定皮肤', 
'mention_data':[{'mention_data': '游戏',
               'offset': 0, 
               'kb_id': '23513',
               'object': '[type]Event[摘要]游戏是一种基于物质需求满足之上的，在一种特定时间、空间范围内遵循某种特定规则的，追求精神需求满足的社会行为方式。游戏有智力游戏,活动性游戏之分，又翻译为Play，Pastime，Playgame，Sport，Spore，Squail，Games，Gamest，Hopscotch，Jeu，Toy。现在的游戏多指各种平台上的电子游戏。[外文名]game，Play，Pastime，Playgame等等[拼音]yóu xì[注音]ㄧㄡˊ ㄒㄧˋ[日文]ゲーム[分类]电子游戏和现实活动性游戏[中文名]游戏[义项描述]娱乐方式[标签]娱乐',
               'confidence': 0.9952694773674011, },
               {'mention_data': '英雄联盟',
               'offset': 3, 
               'kb_id': '161540',
               'object': '[type]Game[摘要]《英雄联盟》(简称LOL)是由美国拳头游戏(Riot Games)开发、中国大陆地区腾讯游戏代理运营的英雄对战MOBA竞技网游。游戏里拥有数百个个性英雄，并拥有排位系统、符文系统等特色养成系统。《英雄联盟》还致力于推动全球电子竞技的发展，除了联动各赛区发展职业联赛、打造电竞体系之外，每年还会举办“季中冠军赛”“全球总决赛”“All Star全明星赛”三大世界级赛事，获得了亿万玩家的喜爱，形成了自己独有的电子竞技文化。[原版名称]League of Legends[制作人]Steven Snow，Travis George[游戏类型]MOBA[游戏平台]Microsoft Windows,Mac OS X[其他名称]撸啊撸、lol[音乐]Christian Linke[编剧]George Krstic[游戏画面]3D[总监]Tom Cadwell，Shawn Carnes[中文名]英雄联盟[玩家人数]多人[发行日期]国服：2011年9月22日,美服：2009年10月27日[开发商]Riot Games[分级]T(ESRB),12(PEGI)[发行商]Riot Games[义项描述]2011年腾讯运营的电子竞技类游戏[标签]娱乐作品[标签]游戏作品[标签]网页游戏[标签]游戏',
               'confidence': 0.9945264458656311, },
               {'mention_data': '皮肤',
               'offset': 14, 
               'kb_id': '44161', 
               'object': '[type]Organization Vocabulary[摘要]皮肤是人体最大的器官，约占体重的16%。皮肤覆盖于全身表面，分为表皮、真皮，并借皮下组织与深部组织相连。皮肤中尚有毛发、皮脂腺、汗腺和指(趾)甲等皮肤附属器。皮肤具有保护、吸收、排泄、感觉、调节体温以及参与物质代谢等作用。[功能]具有重要的屏障保护作用[结构]皮肤由表皮、真皮构成[义项描述]身体表面包在肌肉外面的组织[标签]自然[标签]医学术语',
               'confidence': 0.9793662428855896, }              
   ]
}
