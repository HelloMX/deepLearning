### Tensorflow 常用接口（数据操作）

1. placeholder

   **tf.placeholder(dtype, shape=None, name=None)**

   * dtype：数据类型。常用的是tf.float32,tf.float64等数值类型

   - shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
   - name：名称。

   ```python
   inputX1 = tf.placeholder(dtype=tf.float32,shape=(None,28,28,1))
   inputX2 = tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
   inputy = tf.placeholder(dtype=tf.float32,shape=(None,10))
   ```

2. variable

   **tf.Variable(\<initial - value>，name=\<optional - name>)**

   此函数用于定义图变量。生成一个初始值为initial - value的变量。

   ```
   W=tf.Variable(tf.random_normal(shape))
   ```

   **tf.get_variable(name，shape，dtype，initializer,trainable)**

   此函数用于定义图变量。获取已经存在的变量，如果不存在，就新建一个

   ```python
   W1 = tf.get_variable("W1",[4,4],initializer=tf.contrib.layers.xavier_initializer(seed = 0))

   W2 = tf.get_variable("var", initializer=tf.constant(0.0))
   ```

3. constant

   **tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)**

   ```python
   # Constant 1-D Tensor populated with value list.
   tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

   # Constant 2-D tensor populated with scalar value -1.
   tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                                [-1. -1. -1.]]
   ```

4. matmul

   **tf.matmul(a, b, transpose_a=False, transpose_b=False,adjoint_a=False, adjoint_b=False,   a_is_sparse=False, b_is_sparse=False, name=None)**

   - **a**: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,`complex128` and rank > 1.
   - **b**: `Tensor` with same type and rank as `a`.
   - **transpose_a**: If `True`, `a` is transposed before multiplication.
   - **transpose_b**: If `True`, `b` is transposed before multiplication.
   - **adjoint_a**: If `True`, `a` is conjugated and transposed before multiplication.
   - **adjoint_b**: If `True`, `b` is conjugated and transposed before multiplication.

   ```python
   # 2-D tensor `a`
   # [[1, 2, 3],
   #  [4, 5, 6]]
   a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])

   # 2-D tensor `b`
   # [[ 7,  8],
   #  [ 9, 10],
   #  [11, 12]]
   b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])

   # `a` * `b`
   # [[ 58,  64],
   #  [139, 154]]
   c = tf.matmul(a, b)
   ```

5. 初始化

   **init=tf.global_variables_initializer()**

   ```python
   sess=tf.Session()
   init=tf.global_variables_initializer()
   sess.run(init)
   ```

6. 计算准确率

   **tf.argmax(input, dimension, name=None) **

   * dimension=0 按列找 
   * dimension=1 按行找 
   * tf.argmax()返回最大数值的下标 
   * 通常和tf.equal()一起使用，计算模型准确度

   ```python
   # pred softmax输出（10000,10） pred[0]=[0.1,0.1,0.1,0.1,0.1,0.3,0,0,0,0]
   # y onehot向量（10000,10）  y[0]=[0,0,0,0,0,0,0,0,0,1]
   correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
   accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
   ```

7. 保存，提取参数

   **saver = tf.train.Saver()**

   保存

   ```
   with tf.Session() as sess:
       sess.run(init)
       save_path = saver.save(sess, "my_net/save_net.ckpt")
       print("Save to path: ", save_path)
   ```

   提取

   ```python
   # 先建立 W, b 的容器
   W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
   b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
   # 这里不需要初始化步骤 init= tf.initialize_all_variables()
   saver = tf.train.Saver()
   with tf.Session() as sess:
       # 提取变量
       saver.restore(sess, "my_net/save_net.ckpt")
       print("weights:", sess.run(W))
       print("biases:", sess.run(b))
   ```