"""
    @Time    : 2018/12/10 16:44
    @Author  : DreamMax
    @FileName: cnn_basics.py
    @Software: PyCharm
    @Github  ： https://github.com/HelloMX
"""

import tensorflow as tf


def weight_variable(shape):
    # stddev 为0时效果很差
    W = tf.Variable(tf.random_normal(shape,stddev=0.1))
    # W=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return W


def bias_variable(shape):
    b=tf.Variable(tf.constant(0,dtype=tf.float32,shape=shape))
    return b


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def cnn(inputX,keep_prob):
    # conv1
    W1 = weight_variable([5, 5, 1, 32])
    b1 = bias_variable([32])
    layer1 = conv2d(inputX, W1) + b1
    a_layer1 = tf.nn.relu(layer1)
    p_layer1 = max_pool_2x2(a_layer1)  # 14*14*32
    # conv2
    W2 = weight_variable([5, 5, 32, 64])
    b2 = bias_variable([64])
    layer2 = conv2d(p_layer1, W2) + b2
    a_layer2 = tf.nn.relu(layer2)
    p_layer2 = max_pool_2x2(a_layer2)  # 7*7*64
    # fc
    W_fc_1 = weight_variable([64 * 7 * 7, 1024])
    b_fc_1 = bias_variable([1024])
    flat_layer = tf.reshape(p_layer2, (-1, 64 * 7 * 7))
    layer_fc = tf.matmul(flat_layer, W_fc_1) + b_fc_1
    a_layer_fc = tf.nn.relu(layer_fc)
    drop_a_layer_fc = tf.nn.dropout(a_layer_fc, keep_prob)
    # fc 2
    W_fc_2 = weight_variable([1024, 10])
    b_fc_2 = bias_variable([10])
    layer_fc2 = tf.matmul(drop_a_layer_fc, W_fc_2) + b_fc_2
    prediction = tf.nn.softmax(layer_fc2)
    return prediction,[W1,b1,W2,b2,W_fc_1,b_fc_1,W_fc_2,b_fc_2]


def regression(x):
    x=tf.reshape(x,[-1,784])
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, [W, b]

