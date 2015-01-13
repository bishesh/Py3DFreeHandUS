

from Py3DFreeHandUS.kine import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    
    # Define common variables
    cluster1MkrList = ('Rigid_Body_4-Marker_1','Rigid_Body_4-Marker_2','Rigid_Body_4-Marker_3')
    cluster1Args = {}
    cluster1Args['mkrsLoc'] = createClusterTemplate(readC3D('test_flex2.c3d', ['markers'])['markers'], cluster1MkrList, timeWin=10)
    cluster2MkrList = ('Rigid_Body_3-Marker_1','Rigid_Body_3-Marker_2','Rigid_Body_3-Marker_3')
    cluster2Args = {}
    cluster2Args['mkrsLoc'] = createClusterTemplate(readC3D('test_flex2.c3d', ['markers'])['markers'], cluster2MkrList, timeWin=10)
    stylusArgs = {}
#    stylusArgs['dist'] = [454.0, 365.5, 262.5, 144.0]
#    stylusArgs['markers'] = ('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4')
    stylusArgs['dist'] = [147, 264, 369]
    stylusArgs['markers'] = ('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3')
    stylus = Stylus(fun=collinearNPointsStylusFun, args=stylusArgs)
    
#    def shankPose(mkrs, mkrList, args):
#        
#        return shankPoseISBWithClusterSVD(mkrs, cluster1MkrList, args)
#        
#        
#    def footPose(mkrs, mkrList, args):
#        
#        #return footPoseISBWithClusterSVD(mkrs, cluster2MkrList, args)
#        return calcaneusPoseWithClusterSVD(mkrs, cluster2MkrList, args)
        
    
    def calculateStylusTipInCluster(markers, clusterMkrList, clusterArgs):
        
        # Calculate reference frame
        R, T = rigidBodySVDFun(markers, clusterMkrList, args=clusterArgs)
        
        # Invert roto-translation matrix
        gRl = composeRotoTranslMatrix(R, T)
        lRg = inv2(gRl)
        
        # Reconstruct stylus tip
        stylus.setPointsData(markers)
        stylus.reconstructTip()
        tip = stylus.getTipData()
        
        # Average on the time frames
        tip = np.nanmean(tip, axis=0)
        
        # Add tip to available markers
        markers['Tip'] = tip
        
        # Express tip in the local rigid cluster reference frame
        tipLoc = changeMarkersReferenceFrame(markers, lRg)['Tip']
        
        return tipLoc
    
    
    # --- Cluster 1 ---
    
    # --- LM

    # Read C3D file
    markers = readC3D('LM.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(markers, cluster1MkrList, cluster1Args)
    
    # Calculate average tip in the local reference frame
    LMloc = np.nanmean(tipLoc, axis=0)
    
    # --- MM
    
    # Read C3D file
    markers = readC3D('MM.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(markers, cluster1MkrList, cluster1Args)
    
    # Calculate average tip in the local reference frame
    MMloc = np.nanmean(tipLoc, axis=0)
    
    # --- HF
    
    # Read C3D file
    markers = readC3D('TT.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(markers, cluster1MkrList, cluster1Args)
    
    # Calculate average tip in the local reference frame
    HFloc = np.nanmean(tipLoc, axis=0)
    
    # --- TT
    
    # Read C3D file
    markers = readC3D('HF.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(markers, cluster1MkrList, cluster1Args)
    
    # Calculate average tip in the local reference frame
    TTloc = np.nanmean(tipLoc, axis=0)

    # --- Cluster 2 ---
    
    # --- CA

    # Read C3D file
    markers = readC3D('CA.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(markers, cluster2MkrList, cluster2Args)
    
    # Calculate average tip in the local reference frame
    CAloc = np.nanmean(tipLoc, axis=0)
    
    # --- PT

    # Read C3D file
    markers = readC3D('PT.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(markers, cluster2MkrList, cluster2Args)
    
    # Calculate average tip in the local reference frame
    PTloc = np.nanmean(tipLoc[120:,:], axis=0)
    
    # --- ST

    # Read C3D file
    markers = readC3D('ST.c3d', ['markers'])['markers']
    
    # Express tip in the local rigid cluster reference frame
    tipLoc = calculateStylusTipInCluster(markers, cluster2MkrList, cluster2Args)
    
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
    markers = readC3D('test_flex2.c3d', ['markers'])['markers']
#    markers = readC3D('CA.c3d', ['markers'])['markers']

    # Calculate pose for anatomical reference frames
    R1, T1, markersSeg1 = shankPoseISBWithClusterSVD(markers, cluster1MkrList, segment1Args)
    R2, T2, markersSeg2 = calcaneusPoseWithClusterSVD(markers, cluster2MkrList, segment2Args)
    
    # Calculate joint angles
    angles = getJointAngles(R1, R2)
    
    # Get angles
    FE = angles[:,0]
    AA = angles[:,1]
    IE = angles[:,2]
    
    # Plot FE qngle
    plt.plot(FE)
    plt.show()
    
    # Create dict of calibrated points to write to file
    markersSeg = dict(markersSeg1.items() + markersSeg2.items())
    
    # Write calibrated landmarks to C3D
    data = {}
    data['markers'] = {}
    data['markers']['data'] = markersSeg
    writeC3D('test_flex2_ALs.c3d', data, copyFromFile='test_flex2.c3d')
#    writeC3D('CA_ALs.c3d', data, copyFromFile='CA.c3d')
    
    
    
    
    
    
    
    
    
    
    