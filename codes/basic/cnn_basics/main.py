"""
    @Time    : 2018/12/23 16:13
    @Author  : DreamMax
    @FileName: main.py
    @Software: PyCharm
    @Github  ： https://github.com/HelloMX
"""
import numpy as np

import tensorflow as tf
from flask import Flask,jsonify,render_template,request
from model import regression,cnn
from tensorflow.examples.tutorials.mnist import input_data


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():

    sess = tf.Session()
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    with tf.variable_scope("reg"):
        pred_reg, variables_reg = regression(x);
    saver = tf.train.Saver(variables_reg)
    saver.restore(sess, "my_net/reg_net.ckpt")

    with tf.variable_scope("cnn"):
        keep_prob = tf.placeholder(dtype=tf.float32)
        pred_cnn, variables_cnn = cnn(x, keep_prob)
    saver = tf.train.Saver(variables_cnn)
    saver.restore(sess, "my_net/cnn_net.ckpt")

    def calc_reg(input):
        return sess.run(pred_reg, feed_dict={x: input}).flatten().tolist()

    def calc_cnn(input):
        return sess.run(pred_cnn, feed_dict={x: input, keep_prob: 1}).flatten().tolist()

    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 28, 28, 1)
    output1 = calc_reg(input)
    print(output1)
    output2 = calc_cnn(input)
    print(output2)
    sess.close()
    return jsonify(results=[output1, output2])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == "__main__":
    app.debug = True
    app.run(port=8000)
