

import timeit

setup = """

import numpy as np
from Py3DFreeHandUS import calib_c

h, w = 500, 500
I = np.random.randint(256, size=(200,h,w)).astype(np.uint8)
fr1 = np.arange(100)
fr2 = np.arange(100,150)
mask = np.ones((fr1.shape[0], h, w), dtype=np.uint8)
T = np.random.rand(100*100,4,4)
p1 = np.random.rand(4,h*w)
c1 = np.random.rand(4,h*w).astype(np.int)
thZ = 100
pixel2mmX = 0.01
pixel2mmY = 0.01

"""

number = 100

print 'compoundNCC'
print timeit.timeit("calib_c.compoundNCC(I, mask, fr1, fr2, T, p1, c1, thZ, pixel2mmX, pixel2mmY, 10, '')", setup=setup, number=number)

