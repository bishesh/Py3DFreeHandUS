# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 16:01:32 2014

@author: Davide Monari

Calculate distances for a know object (grid_distances.jpg) by using 2 methods.
"""

from Py3DFreeHandUS.segment import readFeaturesFile, singlePointFeaturesTo3DPointsMatrix
from Py3DFreeHandUS.kine import dot3
import numpy as np

if __name__ == "__main__":

    # ---- METHOD 1: 6-points probing
    
    # Load file containing roto-translation from US image to global
    matrix = np.load('matrices_testwire_3points_10.npz')
    
    # Get roto-translation from US image to global reference frame
    T = matrix['gTim']
    
    # Get single points features from file
    fea = readFeaturesFile('testwire10.fea')   # it must contain one point feature per US image
    
    # Indicate u, v
    u = 30./278
    v = 50./463
    
    # Calculate global coordinates of P1
    idxP1 = [69,70,72,74,79]
    P1 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP1)
    print P1
    P1g = dot3(T[idxP1,:,:], P1[...,None]).squeeze()[:,0:3]  # Np x 3
    P1 = P1g.mean(axis=0)
    print P1g.std(axis=0)
    
    # Calculate global coordinates of P2
    idxP2 = [113,114,115,116,118]
    P2 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP2)
    print P2
    P2g = dot3(T[idxP2,:,:], P2[...,None]).squeeze()[:,0:3]  # Np x 3
    P2 = P2g.mean(axis=0)
    print P2g.std(axis=0)
    
    # Calculate global coordinates for P3
    idxP3 = [144,145,146,147,148]
    P3 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP3)
    print P3
    P3g = dot3(T[idxP3,:,:], P3[...,None]).squeeze()[:,0:3]  # Np x 3
    P3 = P3g.mean(axis=0)
    print P3g.std(axis=0)
    
    # Calculate global coordinates of P4
    idxP4 = [273,274,275,276,279]
    P4 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP4)
    print P4
    P4g = dot3(T[idxP4,:,:], P4[...,None]).squeeze()[:,0:3]  # Np x 3
    P4 = P4g.mean(axis=0)
    print P4g.std(axis=0)
    
    # Calculate global coordinates of P5
    idxP5 = [334,335,336,337,338]
    P5 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP5)
    print P5
    P5g = dot3(T[idxP5,:,:], P5[...,None]).squeeze()[:,0:3]  # Np x 3
    P5 = P5g.mean(axis=0)
    print P5g.std(axis=0)

    # Calculate global coordinates of P6
    idxP6 = [384,385,386,387,388]
    P6 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP6)
    print P6
    P6g = dot3(T[idxP6,:,:], P6[...,None]).squeeze()[:,0:3]  # Np x 3
    P6 = P6g.mean(axis=0)
    print P6g.std(axis=0)
    
    # Calculate distances
    P1_P2 = np.linalg.norm(P1 - P2)
    P2_P3 = np.linalg.norm(P2 - P3)
    P3_P4 = np.linalg.norm(P3 - P4)
    P4_P5 = np.linalg.norm(P4 - P5)
    P5_P6 = np.linalg.norm(P5 - P6)
    P6_P1 = np.linalg.norm(P6 - P1)
    
    print '\n'
    print P1_P2
    print P2_P3
    print P3_P4
    print P4_P5
    print P5_P6
    print P6_P1
    
    # ---- METHOD 2: 6-points manual identification in 3D voxel array
    
    # Load file containing roto-translation from voxel-array to global reference frame
    matrix = np.load('matrices_testwire_3D_14.npz')
    
    # Get roto-translation from US image to global reference frame
    T = matrix['gTva']
    
    # Indicate manually selected in mm (in voxel array reference frame)
    P1 = np.array([86,50,59,1])
    P2 = np.array([85,46,38,1])
    P3 = np.array([81,42,14,1])
    P4 = np.array([140,37,11,1])
    P5 = np.array([143,41,36,1])
    P6 = np.array([146,42,58,1])
    
    # Express points in global reference frame
    P1 = np.dot(T, P1)
    P2 = np.dot(T, P2)
    P3 = np.dot(T, P3)
    P4 = np.dot(T, P4)
    P5 = np.dot(T, P5)
    P6 = np.dot(T, P6)
    
    # Calculate distances
    P1_P2 = np.linalg.norm(P1 - P2)
    P2_P3 = np.linalg.norm(P2 - P3)
    P3_P4 = np.linalg.norm(P3 - P4)
    P4_P5 = np.linalg.norm(P4 - P5)
    P5_P6 = np.linalg.norm(P5 - P6)
    P6_P1 = np.linalg.norm(P6 - P1)
    
    print '\n'
    print P1_P2
    print P2_P3
    print P3_P4
    print P4_P5
    print P5_P6
    print P6_P1
    
    
    
    
    