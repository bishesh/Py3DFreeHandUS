# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:51:02 2014

@author: dmonar0
"""

from Py3DFreeHandUS import process
import numpy as np


if __name__ == "__main__":
   
    
    # --- PROBE CALIBRATION (SPATIAL CALIBRATION)
    
    # Instantiate the Process class for probe calibration
    c = process.Process()
    
    # Set US image properties
    c.setDataSourceProperties(fromUSFiles=('ncc2_dave1_Mevis.dcm',), kineFreq=120, pixel2mmX=50.9/470., pixel2mmY=55.5/512.)
    
    # Set path for US DICOM files
    c.setUSFiles(('ncc2_dave1_Mevis.dcm',)) 
    
    # Set path for markers coordinates file
    c.setKineFiles(('ncc2_dave1.c3d',))
    
    # Set the time delay
    c.setDevicesTimeDelay(0.057)
    
    # And subtract it from the original (make sure to do this before calculating the pose for the US probe)
    c.adjustUSTimeVector()

    # Calculate pose from US probe to laboratory reference frame
    c.calculatePoseForUSProbe(mkrList=('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4'))

    # Extract mask on NCC original images (optional)
    # ATTENTION: selecte frames 16 and 91, not 15 and 90!
    segParams = {}
    segParams['selectorType'] = 'lasso'
    segParams['width'] = 5
    c.extractFeatureFromUSImages(feature='mask', segParams=segParams, featuresFile='mask.fea', showViewer=True)

    # Estimate the pose of the US image plane in the markers-based reference frame on the US probe
    init = {'x1':20.,'y1':-90.,'z1':-20.,'alpha1':np.deg2rad(-10.),'beta1':np.deg2rad(0.),'gamma1':np.deg2rad(-160.)}
    args = {}
    args['sweep_frames'] = [[15, 90],[250,450]]
    args['imag_comp_save_path'] = 'figs'    # make sure this folder already exists
    args['th_z'] = .1
    args['max_expr'] = 'weighted_avg_NCC'
    c.calibrateProbe(init, method='maximize_NCCfast', method_args=args, correctResults=False)
    
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