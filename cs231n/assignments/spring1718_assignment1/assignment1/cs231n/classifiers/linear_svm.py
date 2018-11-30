import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  margins=np.zeros((num_train,num_classes))
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        margins[i,j]=margin
        loss += margin
        
        dW[:,j]+=X[i].T
        dW[:,y[i]]-=X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
#   print(margins[1])
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  dW/=num_train
  dW+=reg*W
  print(dW[1])
  print("loss in naive: "+str(loss))
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_classes = W.shape[1]
  scores=np.dot(X,W)
  n_correct_scores=scores[np.arange(num_train), y]
  n_correct_scores=n_correct_scores.reshape(-1,1)
  margin=scores-n_correct_scores+1
  margin[np.arange(num_train), y]=0
  margin[margin<0]=0
  loss=np.sum(margin)/num_train
    
  loss += 0.5*reg * np.sum(W * W)
  print("loss in vectorized: "+str(loss))
#   print(n_correct_score.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  ones=np.ones(margin.shape)
  ones[margin==0]=0
#   print(margin[1])
  ones[range(num_train),y]=-np.sum(ones,axis=1)
#   print(ones[1])
  dW=np.dot(X.T,ones)/num_train+reg*W
#   print(dW[1])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
