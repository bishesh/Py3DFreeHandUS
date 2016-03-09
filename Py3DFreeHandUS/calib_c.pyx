
import numpy as np
cimport numpy as np

from libc.math cimport abs as abs_c
from libc.math cimport floor as round_c

from libc.stdlib cimport malloc, free

from image_utils_c cimport NCCflat, boolIdx2, rotTrPoint

cimport cython

import matplotlib.pyplot as plt

 
# STRATEGY: the inner loop is huge (about 500 * 500 = 250 000 iterations!), so
# everything inside it must be as fast as possible:
# - use functions inlining
# - use malloc() or fixed_size arrays


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef compoundNCC(np.uint8_t[:,:,:] I, np.uint8_t[:,:,:] mask, int[:] fr1, int[:] fr2, double[:,:,:] T, double[:,:] p1, int[:,:] c1, double thZ, double pixel2mmX, double pixel2mmY, int iterCont, str savePath):
    
    # Instantiate variables
    cdef:
        int N1 = fr1.shape[0]
        int N2 = fr2.shape[0]
        double[:] NCCs = np.empty((N2,), dtype=np.double)
        double[:] pctIs = np.empty((N2,), dtype=np.double)
        int f1, f2, i, j, k, p
        int h = I.shape[1]
        int w = I.shape[2]
        int Np = p1.shape[1]
        double Tk[4][4]
        double p1t[4]
        double p2[4]
        double** I1 = <double**> malloc(h * sizeof(double*))
        double** I2 = <double**> malloc(h * sizeof(double*))
        np.uint8_t** I1mask = <np.uint8_t**> malloc(h * sizeof(np.uint8_t*))
        double** I2rays = <double**> malloc(h * sizeof(double*))
        double[:,:] I1w = np.empty((h,w), dtype=np.double)
        double[:,:] I2w = np.empty((h,w), dtype=np.double)
        np.uint8_t[:,:] I1maskw = np.empty((h,w), dtype=np.uint8)
        double[:,:] I2raysw = np.empty((h,w), dtype=np.double)
        np.uint8_t hasInt
        int c2[2]
        int r, c
        double maskSum, totalSum, pctI, ncc
        
        #np.uint8_t* Iptr = &I[0,0,0]
        #int ptr
        
    try:
#        print np.asarray(fr1)
#        print np.asarray(fr2)
        # Finish creating 2D arrays
        for i in range(h):
            I1[i] = <double*> malloc(w * sizeof(double))
            I2[i] = <double*> malloc(w * sizeof(double))
            I1mask[i] = <np.uint8_t*> malloc(w * sizeof(np.uint8_t))
            I2rays[i] = <double*> malloc(w * sizeof(double))
        
        # Loop for original frames
        for f2 in range(N2):
            #print('f2: %d' % f2)
            # Set I2
            for i in range(h):
                for j in range(w):
                    I1[i][j] = 0
                    I2[i][j] = I[fr2[f2],i,j]
                    I2rays[i][j] = I2[i][j]
                    I1mask[i][j] = 0
            
            # Loop for reconstruction frames
            for f1 in range(N1):
                #print('f1: %d' % f1)
                # Calculate frame number
                k = f2 * N1 + f1
                
                # Extract roto-translation matrix for current frame
                for i in range(4):
                    for j in range(4):
                        Tk[i][j] = T[k,i,j]
                
                # Loop for every pixel
                for p in range(Np):
                    
                    # Extract image point
                    for i in range(4):
                        p1t[i] = p1[i,p]
                    
                    # Calculate coordinates in image 2
                    rotTrPoint(Tk, p1t, p2)
                    
                    if abs_c(p2[2]) < thZ: # mm
                        hasInt = 1
                    else:
                        hasInt = 0
                    
                    if hasInt == 1:                
                        
                        # Convert coordinates from mm to pixels
                        c2[0] = <int>round_c(p2[0] / pixel2mmX)
                        c2[1] = <int>round_c(p2[1] / pixel2mmY)
                        
                        # Create indices for reconstrucete image
                        r = c2[1]
                        c = c2[0]
                        if r < 0:
                            r = 0
                        if r >= h:
                            r = h - 1
                        if c < 0:
                            c = 0
                        if c >= w:
                            c = w - 1
                        
                        # Update reconstruction image
                        I1[r][c] = I[fr1[f1], c1[1,p], c1[0,p]]
                        #I1[r][c] = Iptr[c1[0,p] + c1[1,p] * w + fr1[f1] * (w * h)]
                        
                        # Update reconstruction image mask
                        I1mask[r][c] = mask[f2, r, c]
                        
                        # Add intersections on original image
                        I2rays[r][c] = 255
                
                
#            # Calculate percentage of image information from reconstruction
#            maskSum = 0.
#            for i in range(h):
#                for j in range(w):
#                    if I1mask[i][j] == 1:
#                        maskSum += 1
#            pctI = 100. * maskSum / (h * w)
#            print('Percentage of reconstructed image #%d: %.2f' % (fr2[f2], pctI))
            
            # Element-wise copy is the only way from 2D malloc to typed memoryview
            maskSum = 0.
            totalSum = 0.
            for i in range(h):
                for j in range(w):
                    I1w[i,j] = I1[i][j]
                    I2w[i,j] = I2[i][j]
                    I1maskw[i,j] = I1mask[i][j]
                    I2raysw[i,j] = I2rays[i][j]
                    if I1mask[i][j] > 0:
                        maskSum += 1
                    if mask[f2,i,j] > 0:
                        totalSum += 1
            pctI = 100. * maskSum / totalSum
            print('Percentage of reconstructed image #%d: %.2f' % (fr2[f2], pctI))
                
            # Calculate NCC between the 2 images  (I2 the original, I1 the reconstruction)
            ncc = NCCflat(boolIdx2(I1w, I1maskw), boolIdx2(I2w, I1maskw))
            #ncc = 0
            
            # Save each couple image - reconstruction to file (not cythonizeable)
            if len(savePath) <> 0:
                plt.close('all')
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(np.multiply(I2w, mask[f2,:,:]))  # use np.multiply, not *
                plt.title('Original image (#%d)' % fr2[f2])
                plt.subplot(2,2,2)
                plt.imshow(np.multiply(I1w, mask[f2,:,:]))  # use np.multiply, not *
                plt.title('Reconstruction')
                plt.subplot(2,2,3)
                plt.imshow(I2raysw)
                plt.title('Original image + intersections')
                plt.suptitle('NCC = %.2f' % ncc)
                plt.tight_layout()
                plt.savefig(savePath + '/it%d_im%d.jpeg' % (iterCont,fr2[f2]))
        
            # Store NCCs and reconstruction percentage
            NCCs[f2] = ncc
            pctIs[f2] = pctI
        
    except:
        
        print('Error while computing NCC')
    
    finally:
        
        # Free memory
        for i in range(h):
            free(I1[i])
            free(I2[i])
            free(I1mask[i])
            free(I2rays[i])
        free(I1)
        free(I2)
        free(I1mask)
        free(I2rays)
        
    # Return data
    return np.asarray(NCCs), np.asarray(pctIs)
    
    