"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, flash, request
from RagWeb import app
from RagWeb import myform
import numpy as np
import time
import tensorflow as tf
from gensim.models import word2vec
import jieba
import re
import os
import random

class global_var():
    rag = ""
    dict = [u"离愁", u"伤感", u"豪迈", u"励志", u"孤独", u"闺怨", u"悠闲", u"爱情"]

# 判断感情的函数
def GuessSentiment(instr):
    time_length = 20 # 一首诗歌的词数（多删少补0）
    vec_size = 100 # 一个词的维度
    batch_size = 50

    # 载入word2vec模型
    fname = "word_vec_model"
    model = word2vec.Word2Vec.load(fname)
    vocabulary = model.wv.vocab

    # 诗句预处理（分词）
    s = re.sub(u"[1234567890X]", u"", instr) # 直接省略某些符号
    s = re.sub(u"[，。（）！【】、“”？,.*?/·《》‘’；：|  　]", u" ", s) # 全都变成分隔符
    s = s.replace("\r\n"," ") # 换行变成分隔符，天哪换行回车windows这个编码有毒啊调了半天
    s = jieba.cut(s)
    s1 = u" ".join(s)
    s1 = s1.replace("   "," ") # 不明白分词之后为啥老是出现三个空格，分词把我的空格也当作词语了
    print(u"分词结果:")
    print(s1)
    
    # 诗句转词向量
    test_vec = [model.wv[word] for word in s1.split(" ") if word in vocabulary]
    if np.sum(np.sum(test_vec)) == 0:
        return -1 #  未知
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
        # tf.get_collection() 返回一个list.  但是这里只要第一个参数即可
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
        #predict = y_pre.eval(feed_dict={inputs_: f, keep_prob_:1.0, batch_size_:
        #1})
        print(u"预测结果：%s" % global_var.dict[p[0]])
        return p[0]

@app.route('/')
@app.route('/input')
def input():
    """Renders the home page."""
    # form = myform.message_form()
    # print("Text:", rag)
    return render_template(
        'input.html',
        title='Input',
        year=datetime.now().year,
    )

@app.route('/output', methods=['GET', 'POST'])
def output():
    """Renders the output page.""" 
    music = ["", "", "", "", ""]
    name = [u"未知", u"未知", u"未知", u"未知", u"未知"]

    # 获得诗句
    if request.method == "POST":
        global_var.rag = request.values.get('message')
        print("Get text:", global_var.rag)

    # 判断感情
    if global_var.rag != "":
        result = GuessSentiment(global_var.rag)
        if result == -1:
            message = u"未知"
        else:
            message = global_var.dict[result]
    else:
        message = u"未知"

    # 挑选歌曲
    if message != u"未知":
        pre_path = os.getcwd() + "/RagWeb/static/music/" + message #绝对位置
        pre_path1 = "../static/music/" + message #相对位置
        allfile = os.listdir(pre_path)
        music = []
        name = []
        for i in range(0,5):
            index = random.randint(0, len(allfile)-1)
            name.append(os.path.splitext(allfile[index])[0])
            music.append(pre_path1+"/"+allfile[index])
    
    # 挑选图片
    pic = []
    if message != u"未知":
        img_pre_path = os.getcwd() + "/RagWeb/static/images/image/" + message #绝对位置
        img_pre_path1 = "../static/images/image/" + message #相对位置
    else:
        img_pre_path = os.getcwd() + "/RagWeb/static/images" #绝对位置
        img_pre_path1 = "../static/images" #相对位置

    img_allfile = os.listdir(img_pre_path)
        
    for i in range(0,5):
        img_index = random.randint(0, len(img_allfile)-1)
        pic.append(img_pre_path1+"/"+img_allfile[img_index])


    # 检查
    print(name)
    print(music)

    return render_template(
        'output.html',
        title='Output',
        year=datetime.now().year,
        message = message,
        music = music,
        name = name,
        pic = pic
    )
