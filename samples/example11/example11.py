# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 16:01:32 2014

@author: Francesco Cenni

Calculate muscle-tendon junction -dynamic activity- 
Similar to example10, but richer in data/processing/plotting
"""

from Py3DFreeHandUS.segment import readFeaturesFile, singlePointFeaturesTo3DPointsMatrix
from Py3DFreeHandUS.kine import *
from Py3DFreeHandUS.muscles_analysis import MTJlengths
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    
    # ==== 3-points: Probe as pointer in static & dynamic acquisition

    # Load file containing roto-translation from US image to global
    matrix_pap = np.load('matrices_julie_PaP1.npz')

    # Load file containing roto-translation from US image to global    
    matrix_dynamic = np.load('matrices_julie_dyn3.npz')
    
    # Get roto-translation from US image to global reference frame
    T_pap = matrix_pap['gTim']

    # Get roto-translation from US image to global reference frame
    T_dyn = matrix_dynamic['gTim']
    
    # Get single points features from file
    fea_pap = readFeaturesFile('feature_julie_PaP1.fea')   # it must contain one point feature per US image

    # Get single points features from file     
    fea_dyn = readFeaturesFile('feature_julie_dyn3.fea')   # it must contain one point feature per US image

    # Get metadata from dynamic trial files
    timing_data = np.load('us_time_vector_julie_dyn3.npz')
    usTimeVector = timing_data['usTimeVector'][0]
    fcOpto = 120.    
    
    # Indicate u, v (50mm as depth)
    u = 30./278
    v = 50./463
    
    # Calculate GLOBAL coordinates of muscle insertion - fem - STATIC
    idxP1 = [135,145,155,165,175]
    P1 = singlePointFeaturesTo3DPointsMatrix(fea_pap, u, v, idx=idxP1)
    P1g = dot3(T_pap[idxP1,:,:], P1[...,None]).squeeze()  # Np x 4
    P1 = P1g.mean(axis=0)
    
    # Calculate GLOBAL coordinates of tendon insertion - calc - STATIC
    idxP2 = [285,295,305,315,325]
    P2 = singlePointFeaturesTo3DPointsMatrix(fea_pap, u, v, idx=idxP2)
    P2g = dot3(T_pap[idxP2,:,:], P2[...,None]).squeeze()  # Np x 4
    P2 = P2g.mean(axis=0)
    
    # Calculate GLOBAL coordinates of muscle-tendon junction - STATIC
    idxP3s = [205,215,225,235,230]
    P3s = singlePointFeaturesTo3DPointsMatrix(fea_pap, u, v, idx=idxP3s)
    P3sg = dot3(T_pap[idxP3s,:,:], P3s[...,None]).squeeze()  # Np x 4
    P3s = P3sg.mean(axis=0)
    
    # Calculate GLOBAL coordinates for muscle-tendon junction - DYNAMIC
    #idxP3 = [0,10,20,30,40,45,50,55,60,65,70,80,90,100,105,110,115,120,130,140,150,160,170,180,185,190,195,200,205,210,220,230,240,250,255,260,265,270,275,290,300,310,320,330,340,350,360,370,375,380,385,390,400,410,420,430,435,440,445,450,460,470,480,490,500,510,520,525,530,535,540,545,550,560,570,575,580,585,590,600,610,620,630,640,650,660,670,680,690,695,700,705,710,715,720,725,730,740,750,760,765,770,775,780,785,790,800,810,819] 
    idxP3 = sorted(fea_dyn.keys())    
    P3 = singlePointFeaturesTo3DPointsMatrix(fea_dyn, u, v, idx=idxP3)
    P3g = dot3(T_dyn[idxP3,:,:], P3[...,None]).squeeze()  # Np x 4    
    
    
    
    # ==== Dorsi-Plantar flexion 
    
    # Define common variables: Rigid_Body_2 -> SHANK
    cluster1MkrList = ('shank-Marker_1','shank-Marker_2','shank-Marker_4')
    cluster1Args = {}
    cluster1Args['mkrsLoc'] = createClusterTemplate(readC3D('dyn3.c3d', ['markers'])['markers'], cluster1MkrList, timeWin=[680, 690])
    
    # Define common variables: Rigid_Body_4 -> FOOT
    cluster2MkrList = ('foot-Marker_2','foot-Marker_3','foot-Marker_4')
    cluster2Args = {}
    cluster2Args['mkrsLoc'] = createClusterTemplate(readC3D('dyn3.c3d', ['markers'])['markers'], cluster2MkrList, timeWin=[680, 690])
    stylusArgs = {}
    
    # Pointer
    stylusArgs['dist'] = [149., 266.3, 370.3, 453.2] 
    stylusArgs['markers'] = ('Rigid_Body_4-Marker_1','Rigid_Body_4-Marker_2','Rigid_Body_4-Marker_3','Rigid_Body_4-Marker_4')
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
    markers = readC3D('lat_point.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(stylus, markers, cluster2MkrList, cluster2Args)
    
    # Calculate average tip in the local reference frame
    PTloc = np.nanmean(tipLoc, axis=0)

    # --- ST

    # Read C3D file
    markers = readC3D('med_point.c3d', ['markers'])['markers']
    
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
    segment1Args['side'] = 'R'
    
    segment2Args = cluster2Args.copy()
    segment2Args['mkrsLoc']['CA'] = CAloc
    segment2Args['mkrsLoc']['PT'] = PTloc
    segment2Args['mkrsLoc']['ST'] = STloc
    segment2Args['side'] = 'R'   
    
    # ---- Dynamic trial
    
    # Read dynamic trial
    markers = readC3D('dyn3.c3d', ['markers'])['markers']
    
    # ---- Calculate angles in optoelectronic system timeline
    
    # Calculate pose for anatomical references frame from dynamic trial
    R1d, T1d, markersSeg1 = shankPoseISBWithClusterSVD(markers, cluster1MkrList, segment1Args)
    R2d, T2d, markersSeg2 = calcaneusPoseWithClusterSVD(markers, cluster2MkrList, segment2Args)
    
    # Calculate joint angles
    angles = getJointAngles(R1d, R2d)
    
    # Get angles
    FE = angles[:,0]
    AA = angles[:,1]
    IE = angles[:,2]
    
    # ---- Calculate angles in ultrasound system timeline
    
    # Resample markers
    markers_r, indRes = resampleMarkers(markers, x=usTimeVector, origFreq=fcOpto)
    
    # Calculate pose for anatomical references frame from dynamic trial
    R1d_r, T1d_r, markersSeg1_r = shankPoseISBWithClusterSVD(markers_r, cluster1MkrList, segment1Args)
    R2d_r, T2d_r, markersSeg2_r = calcaneusPoseWithClusterSVD(markers_r, cluster2MkrList, segment2Args)
    
    # Calculate joint angles
    angles_r = getJointAngles(R1d_r, R2d_r)
    
    # Get angles
    FE_r = angles_r[:,0]
    AA_r = angles_r[:,1]
    IE_r = angles_r[:,2]
    

    # ---- Static trial    
    
    # Read static trial
    markers = readC3D('PaP1.c3d', ['markers'])['markers']

    # Calculate pose for anatomical reference frame from static trial
    R1s, T1s, markersSeg1 = shankPoseISBWithClusterSVD(markers, cluster1MkrList, segment1Args)
    
    
    
    # ==== MTJ calculation
    
    # Invert roto-translation matrix
    gRl_1d_r = composeRotoTranslMatrix(R1d_r, T1d_r)
    lRg_1d_r = inv2(gRl_1d_r)
    lRg_1d_r[0,:,:] = np.nan # shank cluster markers are flickering in the beginning
     
    # Invert roto-translation matrix
    gRl_2d_r = composeRotoTranslMatrix(R2d_r, T2d_r)
    lRg_2d_r = inv2(gRl_2d_r)
    
    # Invert roto-translation matrix
    gRl_1s = composeRotoTranslMatrix(R1s, T1s)
    lRg_1s = inv2(gRl_1s)

    # Express points in shank reference frame
    P1shank = np.dot(lRg_1s[100,:,:], P1[...,None]).squeeze()[0:3]
    P2shank = np.dot(lRg_1s[100,:,:], P2[...,None]).squeeze()[0:3]
    P3sshank = np.dot(lRg_1s[100,:,:], P3s[...,None]).squeeze()[0:3]
    P3shank = dot3(lRg_1d_r[idxP3,:,:], P3g[...,None]).squeeze()[:,0:3]  # Np x 3

    # Calculate muscle properties
    Np = P3shank.shape[0]
    DmuscleRaw = np.zeros((Np,))
    DtendonRaw = np.zeros((Np,))
    Dmuscle = np.zeros((Np,))
    Dtendon = np.zeros((Np,))
    
    # Calculate muscle length properties
    ress = MTJlengths(P1shank, P2shank, P3sshank)
    for i in xrange(0, Np):
        res = MTJlengths(P1shank, P2shank, P3shank[i,:])
        DmuscleRaw[i] = res['Dmuscle']
        DtendonRaw[i] = res['Dtendon']
        Dmuscle[i] = res['Dmuscle'] / ress['Dmuscle'] * 100
        Dtendon[i] = res['Dtendon'] / ress['Dtendon'] * 100
    Dcomplex = res['Dcomplex']
    
    
    
    # ==== Data plotting
    
    Nf = FE.shape[0]
    x_t1 = np.arange(Nf) / fcOpto           # opto timeline
    x_t2 = np.array(usTimeVector)[idxP3]    # us timeline
    x_max = np.max([x_t1.max(), x_t2.max()])
    x_min = np.max([x_t1.min(), x_t2.min()])
    
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x_t2, Dmuscle)
    plt.title('Muscle length (% static position)')
    plt.xlim(x_min, x_max)
    
    plt.subplot(3, 1, 2)
    plt.plot(x_t2, Dtendon)
    plt.title('Tendon length (% static position)')
    plt.xlim(x_min, x_max)
    
    plt.subplot(3, 1, 3)
    plt.plot(x_t1, FE)
    plt.hold(True)
    plt.plot(usTimeVector, FE_r, 'r')
    plt.title('FE angle (deg)')
    plt.xlabel('Time (s)')
    plt.xlim(x_min, x_max)
    plt.tight_layout()
    
    DmuscleInterp = interpSignals(x_t2, x_t1, Dmuscle[:,None], kSpline=3).squeeze()
    DtendonInterp = interpSignals(x_t2, x_t1, Dtendon[:,None], kSpline=3).squeeze()
    
    plt.figure()
    plt.plot(x_t2, Dmuscle, 'o', x_t1, DmuscleInterp, '-')
    plt.title('Muscle length')
    plt.xlabel('Time (s)')
    
    plt.figure()
    plt.plot(x_t2, Dtendon, 'o', x_t1, DtendonInterp, '-')
    plt.title('Tendon length')
    plt.xlabel('Time (s)')

    plt.figure()
    plt.plot(FE, DmuscleInterp)
    plt.xlabel('FE angle')
    plt.ylabel('Muscle length')
    
    # ==== Data cutting (FE, DmuscleInterp, DmuscleInterp)
    
    # Define dorsi-flexion edges manually
    edges = [
        [ 2.75,  5.60],
        [12.30, 15.00],
        [24.20, 27.00],
        [34.10, 37.30],
        [45.80, 48.90]
    ]
    
    plt.figure()
    for e in edges:
        # Convert edgess from seconds to frames number
        f1 = np.abs(x_t1 - e[0]).argmin()
        f2 = np.abs(x_t1 - e[1]).argmin()
        # Plot data stripes
        plt.plot(FE[f1:f2], DmuscleInterp[f1:f2])
        plt.hold(True)
        
    plt.xlabel('FE angle')
    plt.ylabel('Muscle length')
    plt.title('Dorsi-flexion only')
        
        
        
    
    # ====
    
    plt.tight_layout()
    plt.show()

