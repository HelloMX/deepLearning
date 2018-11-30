### SVM与SoftMax

1. SVM

   公式：$$ f(x_i, W, b) = W x_i + b $$

   损失函数：$$ L_i = \sum_{j\neq y_i} \max(0, s_j - s_{y_i} + \Delta) ​$$

   梯度的推导：

   ​