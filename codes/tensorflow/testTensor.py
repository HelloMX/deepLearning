import tensorflow as tf

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))#一维 范围是-1 1
WeightsOp=1+Weights


x=tf.placeholder(dtype=float,shape=(1,))
x_feed_in=[1]
k=999
result=k*x+1
grad=tf.gradients(result,x)
# grad1=tf.gradients(result,x_feed_in)   error


init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)


print(sess.run(Weights))
print(sess.run(WeightsOp))

print(sess.run(grad,feed_dict={x:x_feed_in}))
# print(sess.run(grad1,feed_dict={x:x_feed_in}))
