import numpy as np
import time
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as pl
from gensim.models import word2vec
import jieba
import re

time_length = 20 # 一首诗歌的词数（多删少补0）
vec_size = 100 # 一个词的维度
batch_size = 50
dict = [u"离愁", u"伤感", u"豪迈", u"励志", u"孤独", u"闺怨", u"悠闲", u"爱情"]

# 载入word2vec模型
fname = "word_vec_model"
model = word2vec.Word2Vec.load(fname)
vocabulary = model.wv.vocab

# 读取输入的诗句
instr = input(u"输入诗句：")

# 诗句预处理（分词）
s = re.sub(u"[，。（）！【】、“”？,.*?/·X《》‘’；：|1234567890  　]", u"", instr)
s = jieba.cut(s)
instr = u" ".join(s)

# 诗句转词向量
test_vec = [model.wv[word] for word in instr.split(" ") if word in vocabulary]

# 诗句长度标准化
# feature是一个time_length的……所以嗯，如果之后有问题……把它改成二维的长度为1*time_length的？
feature = test_vec[:time_length] if len(test_vec) > time_length else [np.zeros(vec_size)] * (time_length - len(test_vec)) + test_vec
f = np.array([feature])

# 算是测试？
with tf.Session() as sess:
    ## 检查错误
    #print(type(feature))
    #print(type(np.array(feature)))
    #print(type(np.array([feature])))
    
    # 导入模型
    new_saver = tf.train.import_meta_graph("model/poetry_classify_model.meta")
    new_saver.restore(sess, "model/poetry_classify_model")

    # 导入各个……那个叫啥……之前的小变量们？
    # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
    inputs_ = tf.get_collection('inputs')[0]
    labels_ = tf.get_collection('labels')[0]
    keep_prob_ = tf.get_collection('keep_prob')[0]
    batch_size_ = tf.get_collection('batch_size')[0]
    y_pre = tf.get_collection('my_network')[0]
    graph = tf.get_default_graph()

    # 抱歉这几个好像不用哦
    #cross_entropy = tf.get_collection('cross_entropy')[0]
    #accuracy = tf.get_collection('accuracy')

    # 使用y进行预测
    predict = sess.run(y_pre, feed_dict={inputs_: f, keep_prob_: 1.0, batch_size_: 1})
    # 输出预测为1的标签索引位置
    p = sess.run(tf.argmax(predict, 1))

    ## 好像是另一种写法
    #predict = y_pre.eval(feed_dict={inputs_: f, keep_prob_:1.0, batch_size_: 1})
    print(u"预测结果：%s" % dict[p[0]])
