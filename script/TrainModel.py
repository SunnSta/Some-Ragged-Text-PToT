import tensorflow as tf
import time
from gensim.models import word2vec
import numpy as np

# 设置常量
lr = 1e-3
hidden_size = 256
lstm_layers = 3
batch_size = 50 # 注意这个是本次训练传入数据量的常量，而下面的batch_size_是规定神经网络的形状的可变量
time_length = 20 # 一首诗歌的词数（多删少补0）
vec_size = 100 # 一个词的维度
seq_len = time_length * vec_size # 一首诗歌用向量表示的长度
label_size = 10 # onehot表示下的情感标签的维度，也就是输出的长度

# 载入word2vec模型
fname = "word_vec_model"
model = word2vec.Word2Vec.load(fname)
vocabulary = model.wv.vocab

# 载入诗歌数据
infile_t = open("poetry_data_cut3", "r", encoding="utf-8")
infile_l = open("poetry_label", "r", encoding="utf-8")
instr = infile_t.read().split(" \n") # 分成一行行训练数据 # 发现分词以后行末会多个空格，导致split计入空字符
inlabel = infile_l.read().split("\n") # 分成一个个对应标签

# 标签onehot编码
dict = {u"离愁":0, u"伤感":1, u"豪迈":2, u"励志":3, u"孤独":4, u"闺怨":5, u"悠闲":6, u"爱情":7 }
sess0 = tf.InteractiveSession()  # 创建一个新的计算图
label_int = []
for word in inlabel:
    label_int.append(dict[word])

labels = sess0.run(tf.one_hot(label_int, 10))

# 文本转词向量
train_vec = []
for stri in instr:
    # 去掉不在词汇表的词，把每行的字符转向量
    train_vec.append([model.wv[word] for word in stri.split(" ") if word in vocabulary])

features = [i[:time_length] if len(i) > time_length else [np.zeros(vec_size)] * (time_length - len(i)) + i for i in train_vec]
train_size = len(features)

# 把训练数据处理成了:
# features——训练行数（train_size）*每行词数（time_length）*每个词的维度（vec_size）
# labels——训练行数（train_size）*每个标签的维度（label_size）
# 这里打乱一下训练数据的顺序，打乱以后结构和与原来相同，只是行的顺序变了
permutation = np.random.permutation(train_size)
shuffled_dataset = []
for i in permutation:
    shuffled_dataset = shuffled_dataset + [features[i]]

shuffled_labels = labels[permutation]

# 给小变量们（可变参数）（输入数据和对应标签）占个坑，以后训练的时候传入
inputs_ = tf.placeholder(tf.float32, shape=[None, time_length, vec_size], name='inputs')
labels_ = tf.placeholder(tf.float32, shape=[None, label_size], name='labels')
keep_prob_ = tf.placeholder(tf.float32, [], name='keep_prob')
batch_size_ = tf.placeholder(tf.int32, [], name='batch_size') # 每次传入几首诗歌
tf.add_to_collection('batch_size', batch_size_)
tf.add_to_collection('inputs', inputs_)
tf.add_to_collection('labels', labels_)
tf.add_to_collection('keep_prob', keep_prob_)

# 处理输入
inputs = tf.reshape(inputs_, [-1, time_length, vec_size])

# 构建LSTM元胞

## 下面这俩大概是老版本的语法？现在似乎lstm_cell每层都要重新使用函数返回一遍，不然会有奇妙的错误
#lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
#lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
# 难怪照着网上代码一直出错……找到一个能根据版本更新而更新代码的作者真是太感人了┭┮﹏┭┮
## 废弃原代码
#multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(lstm_layers)], state_is_tuple = True)
multi_lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
    cell=tf.contrib.rnn.BasicLSTMCell(hidden_size), input_keep_prob=1.0, output_keep_prob=keep_prob_
    ) for _ in range(lstm_layers)], state_is_tuple = True)
initial_state = multi_lstm_cell.zero_state(batch_size_, tf.float32)

# 展开网络
outputs, state = tf.nn.dynamic_rnn(multi_lstm_cell, inputs=inputs, initial_state=initial_state, time_major=False)
h_state = state[-1][1]


# 用softmax函数得到最后的分类向量……应该是
W = tf.Variable(tf.truncated_normal([hidden_size, label_size], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[label_size]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
tf.add_to_collection('my_network', y_pre)

# 损失和评估函数
cross_entropy = -tf.reduce_mean(labels_ * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(labels_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.add_to_collection('cross_entropy', cross_entropy)
tf.add_to_collection('accuracy', accuracy)

## 开始训练了
## 这个代码为啥一开始训练就有0.9左右的准确率……一定是幻觉
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    t0 = time.time()
#    time0 = time.time()
#    for i in range(train_size - batch_size - 1):
#        X_batch, y_batch = shuffled_dataset[i:i+batch_size], shuffled_labels[i:i+batch_size]
#        cost, acc,  _ = sess.run([cross_entropy, accuracy, train_op], feed_dict={inputs_: X_batch, labels_: y_batch, keep_prob_: 1.0, batch_size_: 50})
#        if (i+1) % 100 == 0:
#            test_acc = 0.0
#            test_cost = 0.0
#            N = 1
#            for j in range(N):
#                X_batch, y_batch = shuffled_dataset[i:i+batch_size], shuffled_labels[i:i+batch_size]
#                _cost, _acc = sess.run([cross_entropy, accuracy], feed_dict={inputs_: X_batch, labels_: y_batch, keep_prob_: 1.0, batch_size_: 50})
#                test_acc += _acc
#                test_cost += _cost
#            print("step {}, train cost={:.6f}, acc={:.6f}; test cost={:.6f}, acc={:.6f}; pass {}s".format(i+1, cost, acc, test_cost/N, test_acc/N, time.time() - time0))
#            time0 = time.time()
#        i = i + batch_size
#    t1 = time.time()

# 随机取出样本的函数
def GetBatch(shuffled_dataset, batch_size, train_size):
    rand_batch = []
    rand_label = []
    for i in range(batch_size):
        r = np.random.randint(0, train_size - batch_size - 1)
        rand_batch.append(shuffled_dataset[r])
        rand_label.append(shuffled_labels[r])
    return rand_batch, rand_label


# 开始训练了(换个代码……试试看……哈哈哈终于看起来正常多了)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t0 = time.time()
    time0 = time.time()
    # 每次batch_size张图片，重复2000次
    for j in range(2000):
        rand_batch, rand_label = GetBatch(shuffled_dataset, batch_size, train_size)
        # 每100次测试一下准确率
        if j%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={inputs_: rand_batch, labels_: rand_label, keep_prob_: 1.0, batch_size_: batch_size})
            print("step %d, training accuracy %g, time %s "%(j, train_accuracy, time.time()-time0))
            time0 = time.time()
        train_op.run(feed_dict={inputs_: rand_batch, labels_: rand_label, keep_prob_: 0.5, batch_size_: 50})
    t1 = time.time()


    #保存训练结果
    saver = tf.train.Saver()  
    saver.save(sess, 'model/poetry_classify_model')
    print("Save done!")

    #输出训练信息
    print("Training Count: ", train_size)
    print("Trainint Time: ",'%d'%((t1-t0)/3600),'Hour ','%d'%((t1-t0)%3600/60),'Min ','%d'%((t1-t0)%60),'Sec')



        