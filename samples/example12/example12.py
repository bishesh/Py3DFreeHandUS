# -*- coding: utf-8 -*-
"""
Created on Tue Dec 09 14:25:45 2014

@author: Davide Monari


"""

from Py3DFreeHandUS.kine import *
import numpy as np

if __name__ == "__main__":

    # List of files to process
    fileNames = [
            'Trial01.c3d',
            'Trial02.c3d',
            'Trial03.c3d',
            'Trial04.c3d',
            'Trial05.c3d',
            'Trial06.c3d',
            'Trial07.c3d',
            'Trial08.c3d',
            'Trial09.c3d',
            'prova3.c3d',
            'prova5.c3d',
            'prova6.c3d',
             ] 

    for fileName in fileNames:
        
        print 'File %s ... ' % fileName
        
        # Calculate stylus tip
        markers = readC3D(fileName, ['markers'])['markers']
        stylusArgs = {}
        stylusArgs['dist'] = [149., 266.3, 370.3, 453.2]
        stylusArgs['markers'] = ('P1','P2','P3','P4')
        stylus = Stylus(P=markers, fun=collinearNPointsStylusFun, args=stylusArgs)
        stylus.reconstructTip()
        stylusTip = stylus.getTipData()
        print 'Stylus tip calculated'
        
        # Write new virtual points to C3D
        data = {}
        data['markers'] = {}
        data['markers']['data'] = {'Tip': stylusTip}
        writeC3D(fileName, data, copyFromFile=fileName)
        print 'New C3D file created'
        
        # Calculate difference between real marker and reconstructed tip
        markerRay = 7.  # mm
        D = np.linalg.norm(stylusTip - markers['M1'], axis=1)
        Dmean = np.mean(np.abs(D - markerRay))
        print 'Distance between reconstructed tip and real tip: %.2f mm' % Dmean
    
