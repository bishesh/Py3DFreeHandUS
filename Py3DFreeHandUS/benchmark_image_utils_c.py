

import timeit

setup = """

import numpy as np
from Py3DFreeHandUS import image_utils_c as itc
from Py3DFreeHandUS import image_utils as it


v1 = np.arange(250000, dtype=np.double)
v2 = np.arange(250000, dtype=np.double)
val = 10
v3 = np.random.rand(500,500)
v4 = np.random.randint(2, size=(500,500)).astype(np.uint8)
v5 = v4.astype(np.bool)
v6 = v3.ravel().astype(np.double)
v7 = np.random.rand(4,4)
v8 = np.random.rand(4,1)
v9 = np.empty((4,1))
"""

number = 5

print 'ewbSum'
print timeit.timeit("itc.ewbSum(v1, val)", setup=setup, number=number)
print timeit.timeit("v1 + val", setup=setup, number=number)

print 'ewbProd'
print timeit.timeit("itc.ewbProd(v1, val)", setup=setup, number=number)
print timeit.timeit("v1 * val", setup=setup, number=number)

print 'ewProd'
print timeit.timeit("itc.ewProd(v1, v2)", setup=setup, number=number)
print timeit.timeit("v1 * v2", setup=setup, number=number)

print 'mean'
print timeit.timeit("itc.mean(v1)", setup=setup, number=number)
print timeit.timeit("np.mean(v1)", setup=setup, number=number)

print 'std'
print timeit.timeit("itc.std(v1)", setup=setup, number=number)
print timeit.timeit("np.std(v1)", setup=setup, number=number)

print 'matProd'
print timeit.timeit("itc.matProd(v7,v8,v9)", setup=setup, number=number)
print timeit.timeit("np.dot(v7,v8)", setup=setup, number=number)

print 'boolIdx2'
print timeit.timeit("itc.boolIdx2(v3, v4)", setup=setup, number=number)
print timeit.timeit("v3[v5]", setup=setup, number=number)

print 'NCC'
print timeit.timeit("itc.NCCflat(v6, v6)", setup=setup, number=number)
print timeit.timeit("it.NCC(v3, v3)", setup=setup, number=number)

