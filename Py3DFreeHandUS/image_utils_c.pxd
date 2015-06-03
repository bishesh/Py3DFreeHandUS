
import numpy as np
cimport numpy as np


cpdef double[:] ewbSum(double[:], double)

cpdef double[:] ewbProd(double[:], double)

cpdef double[:] ewProd(double[:], double[:])

cpdef double mean(double[:])
    
cpdef double std(double[:])

cpdef matProd(double[:,:], double[:,:], double[:,:])

cdef rotTrPoint(double[][4], double[], double[])
    
cpdef double[:] boolIdx2(double[:,:], np.uint8_t[:,:])
    
cpdef double NCCflat(double[:], double[:])

    
    