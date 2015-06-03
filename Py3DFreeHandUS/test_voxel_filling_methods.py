
import numpy as np
import matplotlib.pyplot as plt

V1 = np.zeros((10,), dtype=np.uint8)
V2 = np.zeros(V1.shape, dtype=np.uint8)
contV = np.zeros(V1.shape, dtype=np.uint8)
idxV = np.array([0, 1, 2]).astype(np.int32)
I = np.array([50, 255, 255]).astype(np.uint8)

plt.hold(True)
for i in xrange(200):
    V1[idxV] = (contV[idxV] * V1[idxV]) / (contV[idxV] + 1) + I / (contV[idxV] + 1) # wrong: overflowing
    V2[idxV] = V2[idxV] * (contV[idxV] / (contV[idxV] + 1)) + I * (1. / (contV[idxV] + 1)) # correct
    contV[idxV] += 1
    plt.plot(i, V1[0], 'ro')
    plt.plot(i, V2[0], 'bo')
    
plt.show()