import numpy as np

A=np.array([1,2,3,4])
B=np.random.rand(4)
print(A)
print(A.shape)
print(B.shape)

print("###################")
print(A.T)
print(A.T.shape)
C=np.array([1,2,3,4])
D=np.array([[1,2,3,4]])
print(A*C)
print(A.T*C)
print(A*D)
print(A.T*D)

a=np.random.rand(1,5)
b=np.array([[1,2,3,4,5]])
c=np.array([[1,2,3,4,5]])
print(a)
print(a.shape)
print(b.shape)
print(c.shape)

d=np.random.rand(5,1)
e=np.array([[1],[2],[3],[4],[5]])
print(d)
print(d.shape)
print(e.shape)

print(b*e)
print(e*b)

print(np.dot(b,e))
print(np.dot(e,b))