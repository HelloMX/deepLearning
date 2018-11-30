import numpy as np
import time

a=np.array([1,2,3,4])
print(a)


a=np.random.rand(10000000)
print(a.shape)
b=np.random.rand(10000000)
tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(c)
print("vectorized version: "+str(1000*(toc-tic))+"ms")

c=0
tic=time.time()
for i in range(10000000):
    c+=a[i]*b[i]
toc=time.time()
print(c)
print("For loop: "+str(1000*(toc-tic))+"ms")