"""
    @Time    : 2018/12/23 17:21
    @Author  : DreamMax
    @FileName: train_regression.py
    @Software: PyCharm
    @Github  ： https://github.com/HelloMX
"""
"""
    @Time    : 2018/12/23 17:01
    @Author  : DreamMax
    @FileName: CNN.py
    @Software: PyCharm
    @Github  ： https://github.com/HelloMX
"""

from model import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np



def compute_accuracy(v_xs,v_ys):
    global prediction
    pred=sess.run(prediction,feed_dict={inputX:v_xs})
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={inputX: v_xs, inputy: v_ys})
    return result


mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)

with tf.variable_scope("reg"):
    inputX = tf.placeholder(dtype=tf.float32,shape=(None,28,28,1))
    prediction, variables = regression(inputX)


inputy  = tf.placeholder(dtype=tf.float32,shape=(None,10))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=inputy, logits=prediction))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.02, rho=0.95).minimize(loss)


saver = tf.train.Saver()
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    train_x,train_y=mnist.train.next_batch(100)

    train_x=np.reshape(train_x,(-1,28,28,1))

    l,_=sess.run([loss,optimizer],feed_dict={inputX:train_x,inputy:train_y})

    if i%500==0:
        test_x =mnist.test.images[:1000]
        test_y =mnist.test.labels[:1000]
        test_x= np.reshape(test_x, (-1, 28, 28, 1))
        print("loss: ",l)
        print(compute_accuracy(test_x,test_y))

save_path = saver.save(sess, "my_net/reg_net.ckpt")
print("Save to path: ", save_path)