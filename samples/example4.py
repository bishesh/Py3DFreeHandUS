# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:51:02 2014

@author: dmonar0
"""

from Py3DFreeHandUS import process
from Py3DFreeHandUS.kine import getVersor
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
               
    def globPoseFun(mkrs, mkrList):
        
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
        
    #p.calculatePoseForUSProbe(mkrList=mkrList, globPoseFun=globPoseFun)
    p.calculatePoseForUSProbe(mkrList=mkrList, globPoseFun=None, showMarkers=True)

    # Calculate pose from US images to laboratory reference frame
    p.calculatePoseForUSImages()
    
    # Allocate space for voxel array
    p.initVoxelArray(convR='auto_PCA', fxyz=(1,6,6))
    
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
    p.setVtkImageDataProperties(sxyz=(6,1,1))
    
    # And finally export voxel array to VTI
    p.exportVoxelArrayToVTI('Arthur9.vti')
    
    # Free some memory
    del p
    
    