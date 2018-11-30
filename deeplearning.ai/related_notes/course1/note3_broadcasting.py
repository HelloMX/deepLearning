import numpy as np
A=np.array([[56,0,4.4,68],
            [1.2,104,52,8],
            [1.8,135,99,0.9]])
print(A)
cal=A.sum(axis=0)
print(cal)
percentage=A/cal.reshape(1,4)
# percentage=A/cal  #都可以 reshape用来确保
print(percentage)

B=np.array([[1,2,3,4]])
C=np.array([[5],[6],[7],[8]])
print(B*C)

print(np.dot(B,C))

D=np.array([1,2,3,4])
print(D.shape)
D.reshape((2,2))
print(D.shape)
