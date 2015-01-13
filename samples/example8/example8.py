# -*- coding: utf-8 -*-
"""
Created on Tue Dec 09 14:25:45 2014

@author: Davide Monari


"""

from Py3DFreeHandUS import process
from Py3DFreeHandUS.kine import *
import numpy as np

if __name__ == "__main__":

    # Instantiate the Process class for probe calibration quality check
    q = process.Process()
    
    # Set calibration data
    sx = 30./278
    sy = 50./463
    #x = {'x1':8.4, 'y1':-87.3, 'z1':-19.5, 'alpha1':np.deg2rad(-8.2), 'beta1':np.deg2rad(3.1), 'gamma1':np.deg2rad(-153.9)}
    x = {'x1':7.9, 'y1':-88.4, 'z1':-20.4, 'alpha1':np.deg2rad(-8.0), 'beta1':np.deg2rad(3.0), 'gamma1':np.deg2rad(-154.5)} 
    prRim, Tim = q.evalCalibMatrix(x)
    q.setProbeCalibrationData(prRim, Tim)
    
    # Set US image properties
    q.setDataSourceProperties(fromUSFiles=('pointer_US_4_Mevis.dcm',), kineFreq=120, pixel2mmX=sx, pixel2mmY=sy)
    
    # Set path for US DICOM files
    q.setUSFiles(('pointer_US_4_Mevis.dcm',)) 
    
    # Set path for markers coordinates file
    q.setKineFiles(('pointer_US_4.c3d',))
    
    # Set the time delay
    q.setDevicesTimeDelay(0.057)
    
    # Calculate pose from US probe to laboratory reference frame
    q.calculatePoseForUSProbe(mkrList=('Rigid_Body_2-Marker_1','Rigid_Body_2-Marker_2','Rigid_Body_2-Marker_3','Rigid_Body_2-Marker_4'))
    
    # Set calibration data
    q.setProbeCalibrationData(prRim, Tim)
    
    # Calculate pose from US images to laboratory reference frame
    q.calculatePoseForUSImages()
    
    # Extract the phantom point in the US images
    q.extractFeatureFromUSImages(feature='1_point', segmentation='manual', showViewer=False, featuresFile='points_pointerUS4.fea')
    # Calculate stylus tip
    markers = readC3D('pointer_US_4.c3d', ['markers'])['markers']
    stylusArgs = {}
    #stylusArgs['dist'] = (np.array([147.5, 266., 369., 457.5]) - 3.5).tolist()
    stylusArgs['dist'] = [147, 264, 369, 451]
    stylusArgs['markers'] = ('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4')
    stylus = Stylus(P=markers, fun=collinearNPointsStylusFun, args=stylusArgs)
    stylus.reconstructTip()
    stylusTip = stylus.getTipData()
    
    # Resample tip data
    usTimeVector = np.array(q.getAdjustedUSTimeVector()[0])
    tipRes, ind = resampleMarker(stylusTip, x=usTimeVector, origFreq=120.)

    # Calculate calibration reconstruction accuray (RA)
    q.calculateProbeCalibrationAccuracy(acc='RA', P=tipRes[:ind.max(),:])
    
    # And get it
    listRA, RA = q.getProbeCalibrationAccuracy(acc='RA')
    print 'Reconstruction accuracy for different US probe attitudes: %s' % listRA
    print 'Mean reconstruction accuracy: %.2f mm' % RA
    
    # Write pointer tip to C3D
    data = {}
    data['markers'] = {}
    data['markers']['data'] = {'Tip': stylusTip}
    writeC3D('pointer_US_4_Tip.c3d', data, copyFromFile='pointer_US_4.c3d')