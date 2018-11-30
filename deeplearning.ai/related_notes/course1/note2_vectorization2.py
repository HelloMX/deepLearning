import numpy as np
import math
import time

n=1000000
v=np.random.rand(n)
u_non_vectorized=np.zeros((n,1))


tic=time.time()
for i in range(n):
    u_non_vectorized[i]=math.exp(v[i])
toc=time.time()

print("non_vectorized "+str((toc-tic)*1000)+"ms")

tic=time.time()
u_vectorized=np.exp(v)   #np.ads() ,np.log() .....
toc=time.time()
print("vectorized "+str((toc-tic)*1000)+"ms")