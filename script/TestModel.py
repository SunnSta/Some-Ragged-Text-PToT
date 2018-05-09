import numpy as np
import time
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as pl
import cv2
from tensorflow.examples.tutorials.mnist import input_data

# 和littlecnn一样的命令行参数
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", "input/", "Input directory where train and test datasets are located")
tf.app.flags.DEFINE_string("model_dir", "model/", "Model directory where final model files are saved.")

def main(_):
    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph("model/my_model.meta")
      new_saver.restore(sess, "model/my_model")
      # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
      x = tf.get_collection('x')[0]
      y_ = tf.get_collection('y_')[0]
      keep_prob1 = tf.get_collection('prob1')[0]
      keep_prob2 = tf.get_collection('prob2')[0]
      accuracy = tf.get_collection('accuracy_var')[0]
      y = tf.get_collection('my_network')[0]
      graph = tf.get_default_graph()
  
      # 读取mnist数据
      # minst图片为（n*784）
      mnist = input_data.read_data_sets(FLAGS.input_dir, one_hot=True)

      # 使用y进行预测  
      sess.run(y,feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob1:1.0, keep_prob2:1.0})
      print("test accuracy %g"%accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob1: 1.0, keep_prob2: 1.0}))

if __name__ == "__main__":
  tf.app.run()
