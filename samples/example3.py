# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:51:02 2014

@author: dmonar0
"""

from Py3DFreeHandUS import process
import numpy as np


if __name__ == "__main__":
    

    # --- INTER-DEVICES TIME DELAY ESTIMATION (TEMPORAL CALIBRATION)
    
    # Set source data for 2 acquisitions (more are strongly suggested)
    USFiles = ['test2_temporal_Mevis.dcm','test3_temporal_Mevis.dcm']  
    kineFiles = ['test2_temporal.c3d','test3_temporal.c3d']
    featuresFiles = ['test2_temporal.fea', 'test3_temporal.fea']
    timeDelays = []
    
    for USFile, kineFile, featuresFile in zip(USFiles, kineFiles, featuresFiles):
        
        print USFile, kineFile
    
        # Instantiate the Process class for probe calibration
        c = process.Process()
        
        # Set US image properties
        c.setDataSourceProperties(fromUSFiles=(USFile,), kineFreq=120, pixel2mmX=50.9/470., pixel2mmY=55.5/512.)
        
        # Set path for US DICOM files
        c.setUSFiles((USFile,)) 
        
        # Set path for markers coordinates file
        c.setKineFiles((kineFile,))
    
        # Calculate pose from US probe to laboratory reference frame (here, only traslation will be used later on)
        c.calculatePoseForUSProbe(mkrList=('Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4','Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2'))
        
        # Extract 1 point from the phantom line in the US images
        c.extractFeatureFromUSImages(feature='1_point', segmentation='manual', featuresFile=featuresFile)
        
        # Estimate devices delay
        c.calculateDevicesTimeDelay(method='vert_motion_sync', vertCoordIdx=2, showGraphs=False)
        
        # And get it
        timeDelays.append(c.getDevicesTimeDelay())
        
        # Free some memory
        del c
        
    # Calculate average delay
    timeDelay = np.mean(timeDelays)
    timeDelayStd = np.std(timeDelays)
    print 'Estimated time delay: %.3f +- %.3f s' % (timeDelay, timeDelayStd)
   
    
    # --- PROBE CALIBRATION (SPATIAL CALIBRATION)
    
    # Instantiate the Process class for probe calibration
    c = process.Process()
    
    # Set US image properties
    c.setDataSourceProperties(fromUSFiles=('test2_NCC_Mevis.dcm',), kineFreq=120, pixel2mmX=50.9/470., pixel2mmY=55.5/512.)
    
    # Set path for US DICOM files
    c.setUSFiles(('test2_NCC_Mevis.dcm',)) 
    
    # Set path for markers coordinates file
    c.setKineFiles(('test2_NCC.c3d',))
    
    # Set the time delay
    c.setDevicesTimeDelay(timeDelay)
    
    # And subtract it from the original (make sure to do this before calculating the pose for the US probe)
    c.adjustUSTimeVector()

    # Calculate pose from US probe to laboratory reference frame
    c.calculatePoseForUSProbe(mkrList=('Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4','Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2'))

    # Estimate the pose of the US image plane in the markers-based reference frame on the US probe
    init = {'x1':13.8,'y1':-101.6,'z1':-18.9,'alpha1':np.deg2rad(-11.7),'beta1':np.deg2rad(0.),'gamma1':np.deg2rad(-151.3)}
    args = {}
    args['sweep_frames'] = [[20,70,140],[230,420]]
    args['imag_comp_save_path'] = 'figs'    # make sure this folder already exists
    c.calibrateProbe(init, method='maximize_NCC', method_args=args, correctResults=False)
    
    # And get it
    prRim, Tim, sx, sy, calib = c.getProbeCalibrationData()
    
    # Free some memory
    del c
  
    # --- CALIBRATION ACCURACY ESTIMATION
  
    # ...
    

    # --- CALIBRATION PRECISION ESTIMATION

    # ...
    
    
    # --- VOXEL ARRAY RECONSTRUCTION
    
    # ...