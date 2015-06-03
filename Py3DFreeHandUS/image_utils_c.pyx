
# For some general purpose functions (mean, std, ...), we could have used:
# - bottleneck
# - CythonGSL
# - Ceygen
# but:
# - bottleneck is lightweight but not typed memoryviews templated yet (slower)
# - CythonGSL increases considerably dependencies number (and names are long!)
# - CythonGSL increases considerably dependencies number
#
# Thus, we decided to reimplement all the needed functions with:
# - typed memoryviews support
# - for double types only
# - mostly for 1D arrays only

import numpy as np
cimport numpy as np

from libc.math cimport sqrt as sqrt_c, pow as pow_c

from cpython.array cimport array, clone
from cython.parallel import prange

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] ewbSum(double[:] v, double val):
    
    # Instantiate variables
    cdef:
        int M = v.shape[0]
        double[:] r = np.empty((M,), dtype=np.double)
        #double[:] r = clone(array('d'), M, False)
        int m
    
    for m in range(M):
        r[m] = v[m] + val
        
    return r
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] ewbProd(double[:] v, double val):
    
    # Instantiate variables
    cdef:
        int M = v.shape[0]
        double[:] r = np.empty((M,), dtype=np.double)
        #double[:] r = clone(array('d'), M, False)
        int m
    
    for m in range(M):
        r[m] = v[m] * val
        
    return r 
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] ewProd(double[:] v1, double[:] v2):
    
    # Instantiate variables
    cdef:
        int M1 = v1.shape[0]
        int M2 = v2.shape[0]
        double[:] r
        int m
    
    assert(M1 == M2, 'Non-conformable elements number')    
    
    r = np.empty((M1,), dtype=np.double)
    #r = clone(array('d'), M1, False)
    
    for m in range(M1):
        r[m] = v1[m] * v2[m]
        
    return r
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double mean(double[:] v):
    
    # Instantiate variables
    cdef:
        double s = 0.
        int M = v.shape[0]
        int m
    
    assert(M > 0, 'Empty array')
    
    for m in range(M):
        s += v[m]
    s /= M
    return s
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double std(double[:] v):
    
    # Instantiate variables
    cdef:
        double r, s = 0.
        int M = v.shape[0]
        int m
        double m2
    
    assert(M > 0, 'Empty array')
    
    m2 = mean(v)
    for m in range(M):
        s += pow_c(v[m] - m2, 2)
    r = sqrt_c(s / M)
    return r
    
    
# Adapted from: https://python.g-node.org/python-summerschool-2012/_media/cython/kiel2012_cython.pdf
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline matProd(double[:,:] A, double[:,:] B, double[:,:] out):

    cdef:
        size_t rows_A, cols_A, rows_B, cols_B
        size_t i, j, k
        double s
        
    rows_A, cols_A = A.shape[0], A.shape[1]
    rows_B, cols_B = B.shape[0], B.shape[1]

    # Take each row in A
    for i in range(rows_A):
        # And multiply by every column in B
        for j in range(cols_B):
            s = 0.
            for k in range(cols_A):
                s = s + A[i, k] * B[k, j]
            out[i, j] = s   
            
            

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline rotTrPoint(double A[][4], double b[], double out[]):

    cdef:
        size_t i, k
        double s

    # Take each row in A
    for i in range(4):
        s = 0.
        for k in range(4):
            s = s + A[i][k] * b[k]
        out[i] = s   
    
    
    
    
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] boolIdx2(double[:,:] v, np.uint8_t[:,:] idx):
    
    # Instantiate variables
    cdef:
        int s = 0
        double[:] r
        int M = v.shape[0]
        int N = v.shape[1]
        int Mi = idx.shape[0]
        int Ni = idx.shape[1]
        int m, n
        
    assert(M == Mi, 'Non-conformable rows number')
    assert(N == Ni, 'Non-conformable columns number')
    
    for m in range(M):
        for n in range(N):
            if idx[m, n] <> 0:
                s += 1
                
    #r = clone(array('d'), s, False)
    r = np.empty((s,), dtype=np.double)
    s = 0
    for m in range(M):
        for n in range(N):
            if idx[m, n] == 1:
                r[s] = v[m,n]
                s += 1    
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double NCCflat(double[:] I1, double[:] I2):
    
    # Instantiate variables
    cdef:
        double m1, s1, m2, s2
        double[:] demI1, demI2, prodI, prodIn
        int M1 = I1.shape[0]
        int M2 = I2.shape[0]
        double NCC
        
    assert(M1 == M2, 'Non-conformable elements number')

    # Calculate mean and std for both images
    m1 = mean(I1)
    s1 = std(I1)
    m2 = mean(I2)
    s2 = std(I2)

    # Calculate NCC
    if s1 == 0 or s2 == 0:
        
        # Set it to 0
        NCC = 0.
        
    else:
        
        # Demean images
        demI1 = ewbSum(I1, -m1)
        demI2 = ewbSum(I2, -m2)
        
        # Compute product
        prodI = ewProd(demI1, demI2)
        
        # Normalize it
        prodIn = ewbProd(prodI, 1. / (s1 * s2))
        NCC = mean(prodIn)
        
    return NCC
    
    