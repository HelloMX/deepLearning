### Assignment1总结

1. KNN

   * 基本python数据处理操作

     * 排序: np.argsort()
     * 出现次数：np.bincount()
     * 最大值：np.argmax()

   * 计算L2距离的三种编写方法

     > 测试数据：（500,3072）

     > 训练数据：（5000,3072）

     > 计算每一组测试数据和训练数据之间的L2距离

     * 两次循环

     * 一次循环（向量化）

       ```python
       dists[i,:]=np.sqrt(np.sum((X[i]-self.X_train)**2,axis=1))
       ```

     * 不用循环（差方公式展开，利用广播）

       ```python
       dists+=np.sum(self.X_train**2,1).reshape(1,num_train)
       dists+=np.sum(X**2,1).reshape(num_test,1)
       dists-=2*np.dot(X,self.X_train.T)
       dists=np.sqrt(dists)
       ```

   * 交叉验证