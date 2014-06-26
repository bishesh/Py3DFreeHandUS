# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:51:02 2014

@author: dmonar0
"""

from Py3DFreeHandUS import process
import numpy as np


if __name__ == "__main__":
    
    
    
    # --- PROBE CALIBRATION
    
    # Instantiate the Process class for probe calibration
    c = process.Process()
    
    # Set US image properties
    c.setDataSourceProperties(fromUSFiles=('test3cal_50mm_Mevis.dcm',), kineFreq=120, pixel2mmX=50.9/470., pixel2mmY=55.5/512.)
    
    # Set path for US DICOM files
    c.setUSFiles(('test3cal_50mm_Mevis.dcm',)) 
    
    # Set path for markers coordinates file
    c.setKineFiles(('test3_24mar_50mm.c3d',))

    # Calculate pose from US probe to laboratory reference frame
    c.calculatePoseForUSProbe(mkrList=('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4'))
    
    # Extract 2 points from the phantom line in the US images
    c.extractFeatureFromUSImages(feature='2_points_on_line', segmentation='manual', featuresFile='cal3.fea')    # remove 'featuresFile' argument to pop-up features extraction window on US file
    
    # Estimate: (i) the pose of the US image plane in the markers-based reference frame on the US probe;
    # (ii) the pixel-to-mm factor of the US image (sx, sy)
    # (iii) force to 0 the remaining parameters
    init = {'sx':50.9/470,'sy':55.5/512,'x1':20.,'y1':-90.,'z1':-20.,'alpha1':np.deg2rad(0.),'beta1':np.deg2rad(0.),'gamma1':np.deg2rad(-160.),'z2':650.,'beta2':np.deg2rad(0.),'gamma2':np.deg2rad(20.)}
    c.calibrateProbe(init, method='eq_based', method_args={'phantom':'single_wall','regularize_J':True}, fixed=['sx','sy'], correctResults=False)
    
    # And get it
    prRim, Tim, sx, sy, calib = c.getProbeCalibrationData()
    print 'RMS (mm): %.2f' % calib['RMS']
    print 'Covariance matrix for solution:'
    np.set_printoptions(precision=1)
    np.set_printoptions(suppress=True)
    print calib['root_sol'].cov_x
    
    # Free some memory
    del c
    
    
    
    # --- CALIBRATION ACCURACY ESTIMATION

    # Instantiate the Process class for probe calibration quality check
    q = process.Process()
    
    # Set US image properties
    q.setDataSourceProperties(fromUSFiles=('cube4_Mevis.dcm',), kineFreq=120, pixel2mmX=sx, pixel2mmY=sy)
    
    # Set path for US DICOM files
    q.setUSFiles(('cube4_Mevis.dcm',)) 

    # Set path for markers coordinates file
    q.setKineFiles(('cube4.c3d',))

    # Calculate pose from US probe to laboratory reference frame
    q.calculatePoseForUSProbe(mkrList=('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4'))
    
    # Set calibration data
    q.setProbeCalibrationData(prRim, Tim)
    
    # Calculate pose from US images to laboratory reference frame
    q.calculatePoseForUSImages()
    
    # Extract the phantom point in the US images
    q.extractFeatureFromUSImages(feature='1_point', segmentation='manual', featuresFile='cube4.fea')
    
    # Calculate calibration distance accuray (DA)
    q.calculateProbeCalibrationAccuracy(acc='DA', L=50.)
    
    # And get it
    listDA, DA = q.getProbeCalibrationAccuracy(acc='DA')
    print 'Distance accuracy for different US probe attitudes: %s' % listDA
    print 'Mean distance accuracy: %.2f mm' % DA
    
    

    # --- CALIBRATION PRECISION ESTIMATION

    # Instantiate the Process class for probe calibration quality check
    q = process.Process()
    
    # Set US image properties
    q.setDataSourceProperties(fromUSFiles=('prec7_Mevis.dcm',), kineFreq=120, pixel2mmX=sx, pixel2mmY=sy)
    
    # Set path for US DICOM files
    q.setUSFiles(('prec7_Mevis.dcm',)) 
    
    # Set path for markers coordinates file
    q.setKineFiles(('prec7.c3d',))

    # Calculate pose from US probe to laboratory reference frame
    q.calculatePoseForUSProbe(mkrList=('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4'))
    
    # Set calibration data
    q.setProbeCalibrationData(prRim, Tim)
    
    # Calculate pose from US images to laboratory reference frame
    q.calculatePoseForUSImages()
    
    # Extract the phantom point in the US images
    q.extractFeatureFromUSImages(feature='1_point', segmentation='manual', featuresFile='prec7.fea')
    
    # Calculate calibration reconstruction precision (RP)
    q.calculateProbeCalibrationPrecision(prec='RP')
    
    # And get it
    RPrec = q.getProbeCalibrationPrecision(prec='RP')
    print 'Reconstruction precision: %.2f mm' % RPrec
    
    # Free some memory
    del q
    
    
    # --- VOXEL ARRAY RECONSTRUCTION (ON A LONG AIR-INFLATED BALOON)
    
    # Instantiate the Process class for voxel array reconstruction
    p = process.Process()
    
    # Set calibration data
    p.setProbeCalibrationData(prRim, Tim)
    
    # Set US image properties
    p.setDataSourceProperties(fromUSFiles=('test_baloon2ang_Mevis.dcm',), kineFreq=120, pixel2mmX=sx, pixel2mmY=sy)
    
    # Set path for markers coordinates file
    p.setKineFiles(('test_baloon_2ang.c3d',))

    # Calculate pose from US probe to laboratory reference frame
    p.calculatePoseForUSProbe(mkrList=('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4'))

    # Calculate pose from US images to laboratory reference frame
    p.calculatePoseForUSImages()

    # Reorient global reference frame to be approximately aligned with US scans direction 
    from sympy import Matrix, Symbol, cos as c, sin as s
    alpha = Symbol('alpha')
    beta = Symbol('beta')
    T1 = Matrix(([1,0,0,0],
                 [0,c(alpha),s(alpha),0],
                 [0,-s(alpha),c(alpha),0],
                 [0,0,0,1]
    ))
    T = T1.evalf(subs={'alpha':np.deg2rad(-10.)})
    T = np.array(T).astype(np.float)
    
    # Allocate space for voxel array
    #p.initVoxelArray(convR=np.eye(4), fxyz=(5,1,5))
    p.initVoxelArray(convR=T, fxyz=(1,10,1))
    #p.initVoxelArray(convR='auto_PCA', fxyz=(1,10,10))
    
    # Set path for US DICOM files
    p.setUSFiles(('test_baloon2ang_Mevis.dcm',))
    
    # Set parameters for calculating US images sequence wrapper (or silhouette)
    p.setUSImagesAlignmentParameters(wrapper='convex_hull', step=2)
    
    # Align each US image of each file in the space (will take a bit ...)
    p.alignUSImages()
    
    # Set parameters for gap filling (into the wrapped seauence)
    p.setGapFillingParameters(method='VNN', blocksN=10)
    
    # Fill gaps (go sipping a coffee ...)
    p.fillGaps()
    
    # Set properties for the vtkImageData objects that will exported just below
    p.setVtkImageDataProperties(sxyz=(10,1,10))
    #p.setVtkImageDataProperties(sxyz=(10,1,1))
    
    # Export voxel array silhouette to VTI
    p.exportVoxelArraySilhouetteToVTI('voxel_array_scan_silhouette_baloon.vti')
    
    # And finally export voxel array to VTI
    p.exportVoxelArrayToVTI('voxel_array_baloon.vti')
    
    # Free some memory
    del p

    
    
    # --- VOXEL ARRAY RECONSTRUCTION (LATERAL SCAN OF A HUMAN CALF)
    
    # Instantiate the Process class for voxel array reconstruction
    p = process.Process()
    
    # Set calibration data
    p.setProbeCalibrationData(prRim, Tim)
    
    # Set US image properties
    p.setDataSourceProperties(fromUSFiles=('test4_calf_lat_50mm_Mevis.dcm',), kineFreq=120, pixel2mmX=sx, pixel2mmY=sy)
    
    # Set path for markers coordinates file
    p.setKineFiles(('test_calf_4lat.c3d',))

    # Calculate pose from US probe to laboratory reference frame
    p.calculatePoseForUSProbe(mkrList=('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4'))

    # Calculate pose from US images to laboratory reference frame
    p.calculatePoseForUSImages()

    # Reorient global reference frame to be approximately aligned with US scans direction 
    from sympy import Matrix, Symbol, cos as c, sin as s
    alpha = Symbol('alpha')
    beta = Symbol('beta')
    T1 = Matrix(([1,0,0,0],
                 [0,c(alpha),s(alpha),0],
                 [0,-s(alpha),c(alpha),0],
                 [0,0,0,1]
    ))
    T = T1.evalf(subs={'alpha':np.deg2rad(-30.)})
    T = np.array(T).astype(np.float)
    
    # Allocate space for voxel array
    p.initVoxelArray(convR='auto_PCA', fxyz=(1,8,8))
    #p.initVoxelArray(convR=T, fxyz=(10,1,10))  # in case convR=T
    
    # Set path for US DICOM files
    p.setUSFiles(('test4_calf_lat_50mm_Mevis.dcm',))
    
    # Set parameters for calculating US images sequence wrapper (or silhouette)
    #p.setUSImagesAlignmentParameters(wrapper='convex_hull', step=1)
    
    # Align each US image of each file in the space (will take a bit ...)
    p.alignUSImages()
    
    # Set parameters for gap filling (into the wrapped seauence)
    #p.setGapFillingParameters(method='VNN', blocksN=10)
    
    # Fill gaps (go sipping a coffee ...)
    #p.fillGaps()
    
    # Set properties for the vtkImageData objects that will exported just below
    p.setVtkImageDataProperties(sxyz=(8,1,1))
    #p.setVtkImageDataProperties(sxyz=(1,10,1)) # in case convR=T
    
    # Export voxel array silhouette to VTI
    p.exportVoxelArraySilhouetteToVTI('voxel_array_scan_silhouette_test4_lat.vti')
    
    # And finally export voxel array to VTI
    p.exportVoxelArrayToVTI('voxel_array_test4_lat.vti')
    
    # Free some memory
    del p