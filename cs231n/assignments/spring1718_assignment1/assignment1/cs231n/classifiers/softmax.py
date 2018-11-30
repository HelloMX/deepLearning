import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_class=W.shape[1]
#   print(X[1].shape)
  for i in range(num_train):
    yi=y[i]
    f=np.dot(X[i],W)
    f-=max(f)                 #数值稳定
    
    p = np.exp(f) / np.sum(np.exp(f)) #概率 和为1
    
    loss+=-np.log(p[yi])         #计算损失函数
    for j in range(num_class):
       if j==yi:
        dW[:,j]+=(p[j]-1)*X[i].T
       else:
        dW[:,j]+=p[j]*X[i].T
        
    
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)
  dW=dW/num_train+reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_class=W.shape[1]
  f=np.dot(X,W)
  f -= f.max(axis = 1,keepdims=True)       #(500,10)
  s=np.sum(np.exp(f),axis=1,keepdims=True)   #(500,1)
  p=np.exp(f)/s                     #(500,10)
  pi=p[range(num_train),y].reshape(-1,1)    #(500,1)
  
    
  loss=np.sum(-np.log(pi/s))/num_train+reg*np.sum(W*W)
  p[range(num_train),y]-=1
  dW=np.dot(X.T,p)/num_train+reg*W
#   print(p.shape)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

