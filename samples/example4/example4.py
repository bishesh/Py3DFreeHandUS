# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:51:02 2014

@author: dmonar0
"""

from Py3DFreeHandUS import process
from Py3DFreeHandUS.kine import *
import numpy as np


if __name__ == "__main__":
    
      
    # --- VOXEL ARRAY RECONSTRUCTION (LATERAL SCAN OF A HUMAN CALF)
    
    # Instantiate the Process class for voxel array reconstruction
    p = process.Process()
    
    # Set calibration data
    sx = 0.107914
    sy = 0.107991
    x = {'x1':8.4, 'y1':-87.3, 'z1':-19.5, 'alpha1':np.deg2rad(-8.2), 'beta1':np.deg2rad(3.1), 'gamma1':np.deg2rad(-153.9)}
    prRim, Tim = p.evalCalibMatrix(x)
    p.setProbeCalibrationData(prRim, Tim)
    
    # Set US image properties
    p.setDataSourceProperties(fromUSFiles=('arthur9_Mevis.dcm',), kineFreq=120, pixel2mmX=sx, pixel2mmY=sy)
    
    # Set path for markers coordinates file
    p.setKineFiles(('Arthur9.c3d',))

    # Calculate pose from US probe to laboratory reference frame
    mkrList = ('Rigid_Body_1-Marker_1',
               'Rigid_Body_1-Marker_2',
               'Rigid_Body_1-Marker_3',
               'Rigid_Body_1-Marker_4',
               'Rigid_Body_2-Marker_1',
               'Rigid_Body_2-Marker_2',
               'Rigid_Body_2-Marker_3')
               
    # Probe referene frame function definition
    def rigidProbeClusterFun(mkrs, mkrList, args):
        
        return rigidBodySVDFun(mkrs, mkrList[:4], args)
               
    # Leg reference frame function definition
    def legPoseFun(mkrs, mkrList):
        
        # Define markers to use
        MM = mkrs['Rigid_Body_2-Marker_1']
        LM = mkrs['Rigid_Body_2-Marker_2']
        HF = mkrs['Rigid_Body_2-Marker_3']
        TT = HF
        
        # Create versors        
        O = (LM + MM) / 2
        X = getVersor(np.cross(HF - O, LM - MM))
        Z = getVersor(np.cross(X, TT - O))
        Y = getVersor(np.cross(Z, X))
        
        # Create rotation matrix from global reference frame to laboratory reference frame
        R = np.array((X.T, Y.T, Z.T))   # 3 x 3 x N
        R = np.transpose(R, (2,1,0))  # N x 3 x 3
        
        # Return data
        return R, O
     
    # Coordinates of probe markers in rigid probe reference frame. These have 
    # to be computed from a kienamtic acquisition where markers are well visible
    markersLoc = {}
    markersLoc['Rigid_Body_1-Marker_1'] = np.array([ -7.67213079,  78.5869874 ,   3.87184955])
    markersLoc['Rigid_Body_1-Marker_2'] = np.array([  1.14228084e+02,   6.60250982e+01,  -1.70530257e-13])
    markersLoc['Rigid_Body_1-Marker_3'] = np.array([  1.13686838e-13,   1.13686838e-13,  -1.13686838e-13])
    markersLoc['Rigid_Body_1-Marker_4'] = np.array([  1.06743117e+02,   1.66977543e-13,  -1.42108547e-13])
    args = {}
    args['mkrsLoc'] = markersLoc

    # Declare which US probe reference function to use
    #USProbePoseFun = 'default'
    USProbePoseFun = rigidProbeClusterFun
    
    # Declare which global reference function to use
    #globPoseFun = None
    globPoseFun = legPoseFun
        
    # Calculate pose US probe
    p.calculatePoseForUSProbe(mkrList=mkrList, USProbePoseFun=USProbePoseFun, USProbePoseFunArgs=args, globPoseFun=globPoseFun, showMarkers=False)

    # Calculate pose from US images to laboratory reference frame
    p.calculatePoseForUSImages()
    
    # Set time frames for images that can be cointaned in the voxel array
    p.setValidFramesForVoxelArray()
    
    # Calculate convenient pose for the voxel array
    p.calculateConvPose('auto_PCA')
    
    # Calculate scale factors
    fxyz = 'auto_bounded_parallel_scans'
    #fxyz = (1,6,6)
    p.setScaleFactors(fxyz)
    
    # Calculate voxel array dimensions
    p.calculateVoxelArrayDimensions()
    
    # Allocate memory for voxel array
    p.initVoxelArray()
    
    # Set path for US DICOM files
    p.setUSFiles(('arthur9_Mevis.dcm',))
    
    # Set parameters for calculating US images sequence wrapper (or silhouette)
    #p.setUSImagesAlignmentParameters(wrapper='convex_hull', step=2)
    #p.setUSImagesAlignmentParameters(wrapper='parallelepipedon', step=2)
    p.setUSImagesAlignmentParameters(wrapper='none', step=2)
    
    # Align each US image of each file in the space (will take a bit ...)
    p.alignUSImages()
    
    # Set parameters for gap filling (into the wrapped seauence)
    #p.setGapFillingParameters(method='VNN', blocksN=100, blockDir='X', distTh=None)
    
    # Fill gaps (go sipping a coffee ...)
    #p.fillGaps()
    
    # Set properties for the vtkImageData objects that will exported just below
    sxyz = 'auto'
    #sxyz = (6,1,1)
    p.setVtkImageDataProperties(sxyz=sxyz)
    
    # And finally export voxel array to VTI
    p.exportVoxelArrayToVTI('Arthur9.vti')
    
    # Free some memory
    del p
    
    