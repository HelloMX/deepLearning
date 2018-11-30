import tensorflow as tf
# tf.placeholder()
# tf.get_variable("W1",[],initializer=tf.random_normal_initializer())

import numpy as np
a=np.array([1,2,3,4])
b=a>2
print(b)
print(b.shape)

c=tf.boolean_mask(a,b)
print(c.shape)

sess=tf.Session()
print(c.eval(session=sess))

# with tf.Session() as sess:
#     print(c.eval())