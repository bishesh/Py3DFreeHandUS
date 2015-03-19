# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 16:01:32 2014

@author: Francesco Cenni

Calculate muscle-tendon junction -dynamic activity- 
"""

from Py3DFreeHandUS.segment import readFeaturesFile, singlePointFeaturesTo3DPointsMatrix
from Py3DFreeHandUS.kine import *
from Py3DFreeHandUS.muscles_analysis import MTJlengths
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate


if __name__ == "__main__":

    
    # ==== 3-points: Probe as pointer in static & dynamic acquisition

    # Load file containing roto-translation from US image to global
    matrix_pap = np.load('matrices_test4feb_PaP1.npz')

    # Load file containing roto-translation from US image to global    
    matrix_dynamic = np.load('matrices_test4feb_dyn1.npz')
    
    # Get roto-translation from US image to global reference frame
    T_pap = matrix_pap['gTim']

    # Get roto-translation from US image to global reference frame
    T_dyn = matrix_dynamic['gTim']
    
    # Get single points features from file
    fea_pap = readFeaturesFile('PaP1.fea')   # it must contain one point feature per US image

    # Get single points features from file     
    fea_dyn = readFeaturesFile('dyn1.fea')   # it must contain one point feature per US image
    
    # Indicate u, v (50mm as depth)
    u = 30./278
    v = 50./463
    
    # Calculate GLOBAL coordinates of muscle insertion - fem - STATIC
    idxP1 = [214,216,218,220,222]
    P1 = singlePointFeaturesTo3DPointsMatrix(fea_pap, u, v, idx=idxP1)
    P1g = dot3(T_pap[idxP1,:,:], P1[...,None]).squeeze()  # Np x 4
    P1 = P1g.mean(axis=0)
    
    # Calculate GLOBAL coordinates of tendon insertion - calc - STATIC
    idxP2 = [425,427,429,431,433]
    P2 = singlePointFeaturesTo3DPointsMatrix(fea_pap, u, v, idx=idxP2)
    P2g = dot3(T_pap[idxP2,:,:], P2[...,None]).squeeze()  # Np x 4
    P2 = P2g.mean(axis=0)
    
    # Calculate GLOBAL coordinates for muscle-tendon junction - DYNAMIC
    #idxP3 = [0,20,40,60,70,80,90,100,120,140,160,180,190,199,210,220,240] 
    idxP3 = sorted(fea_dyn.keys())
    P3 = singlePointFeaturesTo3DPointsMatrix(fea_dyn, u, v, idx=idxP3)
    P3g = dot3(T_dyn[idxP3,:,:], P3[...,None]).squeeze()  # Np x 4    
    
    
    
    # ==== Dorsi-Plantar flexion 
    
    # Define common variables: Rigid_Body_2 -> SHANK
    cluster1MkrList = ('Rigid_Body_2-Marker_1','Rigid_Body_2-Marker_2','Rigid_Body_2-Marker_3')
    cluster1Args = {}
    cluster1Args['mkrsLoc'] = createClusterTemplate(readC3D('dyn1pass.c3d', ['markers'])['markers'], cluster1MkrList, timeWin=10)
    
    # Define common variables: Rigid_Body_4 -> FOOT
    cluster2MkrList = ('Rigid_Body_4-Marker_1','Rigid_Body_4-Marker_2','Rigid_Body_4-Marker_3')
    cluster2Args = {}
    cluster2Args['mkrsLoc'] = createClusterTemplate(readC3D('dyn1pass.c3d', ['markers'])['markers'], cluster2MkrList, timeWin=10)
    stylusArgs = {}
    
    # Pointer
    stylusArgs['dist'] = [147.4, 263.4, 367.1, 449.4]
    stylusArgs['markers'] = ('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4')
    stylus = Stylus(fun=collinearNPointsStylusFun, args=stylusArgs)
 
 
    # --- Cluster 1 --- SHANK
    
    # --- LM

    # Read C3D file
    markers = readC3D('LM.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(stylus, markers, cluster1MkrList, cluster1Args)
    
    # Calculate average tip in the local reference frame
    LMloc = np.nanmean(tipLoc, axis=0)
    
    # --- MM
    
    # Read C3D file
    markers = readC3D('MM.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(stylus, markers, cluster1MkrList, cluster1Args)
    
    # Calculate average tip in the local reference frame
    MMloc = np.nanmean(tipLoc, axis=0)
    
    # --- HF
    
    # Read C3D file
    markers = readC3D('TT.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(stylus, markers, cluster1MkrList, cluster1Args)
    
    # Calculate average tip in the local reference frame
    HFloc = np.nanmean(tipLoc, axis=0)
    
    # --- TT
    
    # Read C3D file
    markers = readC3D('HF.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(stylus, markers, cluster1MkrList, cluster1Args)
    
    # Calculate average tip in the local reference frame
    TTloc = np.nanmean(tipLoc, axis=0)

    
    # --- Cluster 2 --- FOOT
    
    # --- CA

    # Read C3D file
    markers = readC3D('CA.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(stylus, markers, cluster2MkrList, cluster2Args)
    
    # Calculate average tip in the local reference frame
    CAloc = np.nanmean(tipLoc, axis=0)
    
    # --- PT

    # Read C3D file
    markers = readC3D('PT.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(stylus, markers, cluster2MkrList, cluster2Args)
    
    # Calculate average tip in the local reference frame
    PTloc = np.nanmean(tipLoc, axis=0)
    
    # --- ST

    # Read C3D file
    markers = readC3D('ST.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(stylus, markers, cluster2MkrList, cluster2Args)
    
    # Calculate average tip in the local reference frame
    STloc = np.nanmean(tipLoc, axis=0)

    # Set parameters for anatomical reference frame construction
    segment1Args = cluster1Args.copy()
    segment1Args['mkrsLoc']['LM'] = LMloc
    segment1Args['mkrsLoc']['MM'] = MMloc
    segment1Args['mkrsLoc']['HF'] = HFloc
    segment1Args['mkrsLoc']['TT'] = TTloc
    segment1Args['side'] = 'L'
    
    segment2Args = cluster2Args.copy()
    segment2Args['mkrsLoc']['CA'] = CAloc
    segment2Args['mkrsLoc']['PT'] = PTloc
    segment2Args['mkrsLoc']['ST'] = STloc
    segment2Args['side'] = 'L'   
    
    # Read dynamic trial
    markers = readC3D('dyn1pass.c3d', ['markers'])['markers']
    
    # Calculate pose for anatomical references frame from dynamic trial
    R1d, T1d, markersSeg1 = shankPoseISBWithClusterSVD(markers, cluster1MkrList, segment1Args)
    R2d, T2d, markersSeg2 = calcaneusPoseWithClusterSVD(markers, cluster2MkrList, segment2Args)
    
    # Read static trial
    markers = readC3D('PaP_1.c3d', ['markers'])['markers']

    # Calculate pose for anatomical reference frame from static trial
    R1s, T1s, markersSeg1 = shankPoseISBWithClusterSVD(markers, cluster1MkrList, segment1Args)
    
    # Calculate joint angles
    angles = getJointAngles(R1d, R2d)
    
    # Get angles
    FE = angles[:,0]
    AA = angles[:,1]
    IE = angles[:,2]
    FE[0:2] = np.nan    # first 2 points have an issue here  
    
    
    
    # ==== MTJ calculation
    
    # Invert roto-translation matrix
    gRl_1d = composeRotoTranslMatrix(R1d, T1d)
    lRg_1d = inv2(gRl_1d)
     
    # Invert roto-translation matrix
    gRl_2d = composeRotoTranslMatrix(R2d, T2d)
    lRg_2d = inv2(gRl_2d)
    
    # Invert roto-translation matrix
    gRl_1s = composeRotoTranslMatrix(R1s, T1s)
    lRg_1s = inv2(gRl_1s)

    # Express points in shank reference frame
    fcUS = 15.
    fcOpto = 120.
    fcRatio = fcOpto / fcUS
    idx = (np.array(idxP3) * fcRatio).astype(np.int32)
    P1shank = np.dot(lRg_1s[100,:,:], P1[...,None]).squeeze()[0:3]
    P2shank = np.dot(lRg_1s[100,:,:], P2[...,None]).squeeze()[0:3]
    P3shank = dot3(lRg_1d[idx,:,:], P3g[...,None]).squeeze()[:,0:3]  # Np x 3

    # Calculate muscle properties
    Np = P3shank.shape[0]
    Dmuscle = np.zeros((Np,))
    Dtendon = np.zeros((Np,))
    
    # Calculate muscle length properties
    for i in xrange(0, Np):
        res = MTJlengths(P1shank, P2shank, P3shank[i,:])
        Dmuscle[i] = res['Dmuscle']
        Dtendon[i] = res['Dtendon']
    Dcomplex = res['Dcomplex']
    
    
    
    # ==== Data plotting
    
    Nf = FE.shape[0]
    x_t1 = np.arange(Nf) / fcOpto
    x_t2 = np.array(idxP3) / fcUS
    x_max = np.max([x_t1.max(), x_t2.max()])
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x_t2, Dmuscle)
    plt.title('Muscle length')
    plt.xlim(0, x_max)
    
    plt.subplot(3, 1, 2)
    plt.plot(x_t2, Dtendon)
    plt.title('Tendon length')
    plt.xlim(0, x_max)
    
    plt.subplot(3, 1, 3)
    plt.plot(x_t1, FE)
    plt.title('FE angle')
    plt.xlabel('Time (s)')
    plt.xlim(0, x_max)
    plt.tight_layout()
    
    fInterp = interpolate.interp1d(x_t2, Dmuscle, bounds_error=False, kind='cubic')
    DmuscleInterp = fInterp(x_t1)
    plt.figure()
    plt.plot(x_t2, Dmuscle , 'o', x_t1, DmuscleInterp, '-')
    plt.title('Muscle length')
    plt.xlabel('Time (s)')

    plt.figure()
    plt.plot(FE, DmuscleInterp)
    plt.xlabel('FE angle')
    plt.ylabel('Muscle length')
    
    plt.tight_layout()
    plt.show()

