# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 16:01:32 2014

@author: Davide Monari

Calculate muscle-tendon junction related parameters by using 3 methods.
"""

from Py3DFreeHandUS.segment import readFeaturesFile, singlePointFeaturesTo3DPointsMatrix
from Py3DFreeHandUS.kine import dot3
from Py3DFreeHandUS.muscles_analysis import MTJlengths
import numpy as np

if __name__ == "__main__":

    # ---- METHOD 1: 3-points probing

    # Load file containing roto-translation from US image to global
    matrix = np.load('matrices_plantar_3points.npz')
    
    # Get roto-translation from US image to global reference frame
    T = matrix['gTim']
    
    # Get single points features from file
    fea = readFeaturesFile('plantar_position1.fea')   # it must contain one point feature per US image
    
    # Indicate u, v
    u = 30./278
    v = 50./463
    
    # Calculate global coordinates of muscle insertion
    idxP1 = [3,5,8,12,16]
    #idxP1 = [510,512,515,518,521]
    P1 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP1)
    P1g = dot3(T[idxP1,:,:], P1[...,None]).squeeze()[:,0:3]  # Np x 3
    P1 = P1g.mean(axis=0)
    
    # Calculate global coordinates of tendon insertion
    idxP2 = [510,512,515,518,521]
    #idxP2 = [3,5,8,12,16]
    P2 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP2)
    P2g = dot3(T[idxP2,:,:], P2[...,None]).squeeze()[:,0:3]  # Np x 3
    P2 = P2g.mean(axis=0)
    
    # Calculate global coordinates for muscle-tendon junction
    idxP3 = [146,149,154,156,159]
    P3 = singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=idxP3)
    P3g = dot3(T[idxP3,:,:], P3[...,None]).squeeze()[:,0:3]  # Np x 3
    P3 = P3g.mean(axis=0)
    
    # Calculate muscle length properties
    res1 = MTJlengths(P1, P2, P3)
    print res1
    
    
    
    # ---- METHOD 2: 3-points manual identification in 3D voxel array
    
    # Load file containing roto-translation from voxel-array to global reference frame
    matrix = np.load('matrices_plantar_3D.npz')
    
    # Get roto-translation from US image to global reference frame
    T = matrix['gTva']
    
    # Indicate manually selected in mm (in voxel array reference frame)
    P1 = np.array([402, 47, 46, 1])
    P2 = np.array([16, 54, 51, 1])
    P3 = np.array([166, 59, 41, 1])
    
    # Express points in global reference frame
    P1 = np.dot(T, P1)
    P2 = np.dot(T, P2)
    P3 = np.dot(T, P3)
    
    # Calculate muscle length properties
    res2 = MTJlengths(P1, P2, P3)
    print res2
    
    
    
    # ---- METHOD 3: manual scale measures
    
    # Indicate manual scale measures
    Dtendon = 145.
    Dmuscle = 252.
    Dcomplex = 397.
    
     # Calculate muscle length properties
    res3 = {}
    res3['Dtendon'] = Dtendon
    res3['Dmuscle'] = Dmuscle
    res3['Dcomplex'] = Dcomplex
    res3['DmusclePct'] = Dmuscle / (Dtendon + Dmuscle)
    res3['DtendonPct'] = Dtendon / (Dtendon + Dmuscle)
    print res3
    
    
    