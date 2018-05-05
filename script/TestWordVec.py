# -*- coding: utf-8 -*-
from gensim.models import word2vec

#载入模型
fname = "word_vec_model"
model = word2vec.Word2Vec.load(fname)

# 计算两个词的相似度/相关程度
a = u"爱"
b = u"面"
try:
    y1 = model.similarity(a, b)
except keyerror:  
    y1 = 0  
print(a, b, u"相似度：", y1)  
print("-----\n")

# 计算某个词的相关词列表  
c = u"看"
y2 = model.most_similar(c, topn=20)  # 20个最相关的  
print(c, u"最相关的20个词：\n")
for item in y2:  
    print(item[0], item[1])  
print("-----\n")
   
# 寻找不合群的词  
d = u"战 乱 悲 核"
y4 =model.doesnt_match(d.split())  
print(d, u"不合群的词：", y4)
print("-----\n")

# 输出词向量
print(model.wv['风'])
