


import numpy as np
from Py3DFreeHandUS import image_utils_c as itc
from Py3DFreeHandUS import image_utils as it


v1 = np.random.rand(250000)
v2 = np.random.rand(250000)
val = 10
v3 = np.random.rand(500,500)
v4 = np.random.randint(2, size=(500,500)).astype(np.uint8)
v5 = v4.astype(np.bool)
v6 = v3.ravel().astype(np.double)
v7 = np.random.rand(4,4)
v8 = np.random.rand(4,1)
v9 = np.empty((4,1))


print 'ewbSum'
print np.asarray(itc.ewbSum(v1, val))
print v1 + val

print 'ewbProd'
print np.asarray(itc.ewbProd(v1, val))
print v1 * val

print 'ewProd'
print np.asarray(itc.ewProd(v1, v2))
print v1 * v2

print 'mean'
print itc.mean(v1)
print np.mean(v1)

print 'std'
print itc.std(v1)
print np.std(v1)

print 'matProd'
print np.asarray(itc.matProd(v7,v8,v9))
print np.dot(v7,v8)

print 'boolIdx2'
print np.asarray(itc.boolIdx2(v3, v4))
print v3[v5]

print 'NCC'
print itc.NCCflat(v6, v6)
print it.NCC(v6, v6)

