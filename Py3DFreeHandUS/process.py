# -*- coding: utf-8 -*-
"""
.. module:: process
   :synopsis: Module for performing: US probe calibration; calibration quality assessment; voxel-array reconstruction.

"""


# Modules importation
from kine import *
from voxel_array_utils import *
from image_utils import *
from segment import *
from calib import *
from array_helpers import *
from math_utils import lcmm
import numpy as np
from scipy import ndimage as nd
import time
import vtk
from itertools import combinations
import copy
import os
from matplotlib import pyplot as plt



def checkOdd(n):
    if n % 2 <> 0:
        return True
    return False
    
def checkInt(n):
    if abs(round(n)-n) == 0:
        return True
    return False

def checkFreq(f):
    if f == None:
        raise Exception('Acquisition frequency was not defined')
#    if not checkInt(f) or f <= 0:
#        raise Exception('Acquisition frequency must be integer and positive')

def checkFreqRatio(f1, f2):
    if not checkInt(f2/f1) and not checkInt(f2/f1):
        raise Exception('Frequencies ratio must be integer')

def checkMkrList(mkrList):
    if len(set(mkrList)) < 3:
        raise Exception('There must be at least 3 markers')

def checkKineFiles(kineFiles, L=None):
    if kineFiles == None:
        raise Exception('Kinematics files were not set')
    if L <> None and len(kineFiles) <> L:
        raise Exception('Number of kinematics files must be {0}'.format(L))

def checkUsFiles(usFiles, L=None):
    if usFiles == None:
        raise Exception('US files were not set')
    if L <> None and len(usFiles) <> L:
        raise Exception('Number of US files must be {0}'.format(L))


def checkIm2PrPose(prRim, Tim):
    if prRim == None or Tim == None:
         raise Exception('US probe calibration was not performed')        
    
    if len(prRim.shape) <> 2 or prRim.shape[0] <> 3 or prRim.shape[1] <> 3:
        raise Exception('US image-to-probe rotation matrix must be a 3 x 3 matrix')
    
    if len(Tim.shape) <> 1 or Tim.shape[0] <> 3:
        raise Exception('US image-to-probe position vector must be a 3 elements vector')

def checkPr2GlPose(Rpr, Tpr):
    if Rpr == None or Tpr == None:
         raise Exception('US probe pose computation was not performed')

    if len(Rpr.shape) <> 3 or Rpr.shape[1] <> 3 or Rpr.shape[2] <> 3:
        raise Exception('Probe-to-global rotation matrix must be a N x 3 x 3 matrix')

    if len(Tpr.shape) <> 2 or Tpr.shape[1] <> 3:
        raise Exception('Probe-to-global position vector must be a N x 3 matrix')

def checkIm2GlPose(R):
    if R == None:
        raise Exception('Pose for US images was not calculated')
    if len(R.shape) <> 3 or R.shape[1] <> 4 or R.shape[2] <> 4:
        raise Exception('Image-to-global roto-translation matrix must be a N x 4 x 4 matrix')

def checkGl2ConvPose(R):
    if R == None:
        raise Exception('Pose from global to convenient reference frame to was not set')
    if isinstance(R, basestring):
        if R not in ['auto_PCA','first_last_frames_centroid']:
            raise Exception(' global-to-convenient roto-translation matrix calculation method not supported')
        return
    if len(R.shape) <> 2 or R.shape[0] <> 4 or R.shape[1] <> 4:
        raise Exception('If matrix, global-to-convenient roto-translation matrix must be 4 x 4 matrix')

def checkFeature(feature):
    if feature not in ['2_points_on_line', '2_points', '1_point', 'mask']:
        raise Exception('Feature not supported')

def checkSegmentation(segmentation):
    if segmentation not in ['manual','auto_hough']:
        raise Exception('Segmentation not supported')

def checkPhantom(phantom):
    if phantom not in ['single_wall']:
        raise Exception('Phantom not supported')

def checkFeatures(features):
    if features == None or len(features) == 0:
        raise Exception('Features from US images were not extracted')

def checkImDim(d):
    if d == None:
        raise Exception('At least one US image dimensions was not set')
    if not checkInt(d) or d <= 0:
        raise Exception('US image dimensions must be integer and positive')

def checkPixel2mm(pixel2mm):
    if pixel2mm == None:
        raise Exception('US image pixel-to-mm ratio was not set')
    if pixel2mm <= 0:
        raise Exception('US image pixel-to-mm ratio must be positive')

def checkFxyz(fxyz):
    if fxyz == None:
        raise Exception('Voxel array scaling factors were not set')
    if isinstance(fxyz, basestring):
        if fxyz not in ['auto_bounded_parallel_scans']:
            raise Exception('Voxel array scaling factors calculation method not supported')
        return
    if len(fxyz) <> 3:
        raise Exception('Voxel array scaling factors must be exactly 3')
    for i in xrange(0,3):
        if fxyz[i] <= 0:
            raise Exception('All voxel array scaling factors must be positive')  

def checkWrapper(wrapper):
    if wrapper == None:
        raise Exception('Wrapping method was not set')
    if wrapper not in ['parallelepipedon', 'convex_hull','none']:
        raise Exception('Wrapping method not supported')

def checkStep(step):
    if step == None:
        raise Exception('Wrapping creation step was not set')
    if not checkInt(step) or step <= 0:
        raise Exception('Wrapping creation step must be integer and positive')

def checkV(V):
    if V == None:
        raise Exception('Voxel array initialization was not performed')
        
def checkPathForSuppFiles(fp):
    if fp == None:
        raise Exception('Path for support files was not set')
    if not os.path.isdir(fp):
        raise Exception('Path for support files is not valid')

def checkMethod(method):
    if method == None:
        raise Exception('Gaps filling method was not set')
    if method not in ['VNN', 'AVG_CUBE']:
        raise Exception('Gaps filling method not supported')

def checkBlocksN(blocksN):
    if blocksN == None:
        raise Exception('Blocks number was not set')
    if not checkInt(blocksN) or blocksN <= 0:
        raise Exception('Blocks number must be integer and positive or zero')
        
def checkBlockDir(d):
    if d == None:
        raise Exception('Blocks direction was not set')
    if d not in ['X', 'Y', 'Z']:
        raise Exception('Blocks direction not supported')

def checkMaxS(maxS):
    if maxS == None:
        raise Exception('Max search cube side was not set')
    if not checkInt(maxS) or not checkOdd(maxS) or maxS <= 0:
        raise Exception('Max search cube side must be integer, positive and odd')

def checkDistTh(d):
    if d <> None and d < 1:
        raise Exception('Distance threshold must be greater or equal than 1')

def checkMinPct(minPct):
    if minPct == None:
        raise Exception('Acceptability percentage was not set')            
    if minPct < 0:
        raise Exception('Acceptability percentage must be positive or zero')

def checkSxyz(sxyz):
    if sxyz == None:
        raise Exception('vtkImageData spacing factors were not set')
    if isinstance(sxyz, basestring):
        if sxyz not in ['auto']:
            raise Exception('vtkImageData spacing factors calculation method not supported')
        return
    if len(sxyz) <> 3:
        raise Exception('vtkImageData spacing factors must be exactly 3')
    for i in xrange(0,3):
        if sxyz[i] <= 0:
            raise Exception('All vtkImageData spacing factors must be positive')
            
def checkFilePath(p):
    if p == None:
        raise Exception('File path was not set')
    if len(p) == 0:
        raise Exception('File path cannot be empty')

def checkPrecType(p):
    if p == None:
        raise Exception('Precision type was not set')
    if p not in ['RP']:
        raise Exception('Precision type not supported')

def checkAccType(a):
    if a == None:
        raise Exception('Accuracy type was not set')
    if a not in ['DA', 'RA']:
        raise Exception('Accuracy type not supported')

def checkDist(d):
    if d == None:
        raise Exception('Distance was not set')
    if not checkInt(d) or d <= 0:
        raise Exception('Distance must be integer and positive')

def checkTimeVector(t):
    if t == None:
        raise Exception('Time vector was not set')  
    if len(t) == 0:
        raise Exception('Time vector cannot be empty') 
#    if t[0] <> 0:
#        raise Exception('First time element must be 0')
        
def checkTimeDelay(t):
    if t == None:
        raise Exception('Time delay was not set')     


def setInsideRange(v, bound, stepBase):
    while True:
        if v <= bound and v >= -bound:
            break
        step = -np.sign(v) * stepBase
        v += step
    return v
    

def checkCalibMethod(method):
    if method == None:
        raise Exception('Calibation method was not set')
    if method not in ['eq_based', 'maximize_NCCint', 'maximize_NCC', 'maximize_NCCfast']:
        raise Exception('Calibration method not supported')

        
def checkAlignFrames(alignFrames, N):
    if alignFrames == None:
        raise Exception('Frames for alignment were not set')
    if min(alignFrames) < 0 or max(alignFrames) > N-1:
        raise Exception('Some frame for alignment out of bounds')


def checkFillVoxMethod(method):
    if method == None:
        raise Exception('Voxel filling method was not set')
    if method not in ['last', 'avg', 'max']:
        raise Exception('Voxel filling method not supported')

        
def checkVoxFrames(voxFrames, N):
    if voxFrames == None:
        raise Exception('Frames for voxel array reconstruction were not set')
    if isinstance(voxFrames, basestring):
        if voxFrames not in ['all','auto']:
            pass
        return
    if min(voxFrames) < 0 or max(voxFrames) > N-1:
        raise Exception('One frame for voxel array reconstruction out of bounds')
        
  
def checkVoxFramesBounds(voxFramesBounds, N):
    if voxFramesBounds <> None:
        if voxFramesBounds[0] < 0 or voxFramesBounds[1] > N-1:
            raise Exception('Frame bounds for voxel array reconstruction lesser than 0 or bigger than %d' % N)
      
        
def checkTemporalCalibMethod(method):
    if method == None:
        raise Exception('Temporal calibration method was not set')
    if method not in ['vert_motion_sync']:
        raise Exception('Temporal calibration method not supported')

# Process class


class Process:
    """Class for performing: US probe calibration; calibration quality assessment; voxel-array reconstruction
    """
    
    def __init__(self):
        """Constructor
        """

        # Data source files
        self.kineFiles = None
        self.usFiles = None
        
        # US images parameters
        self.w = None
        self.h = None
        self.pixel2mmX = None
        self.pixel2mmY = None
        self.usFreq = None
        self.usTimeVector = None
        self.usTimeVectorAdj = None
        
        # Kinematics file properties
        self.kineFreq = None
        
        # US probe-to-lab attitube
        self.Rpr = None
        self.Tpr = None
        
        # Image-to-US probe attitude
        self.prRim = None
        self.Tim = None
        
        # Calibration results
        self.calib = None
        
        # Image-to-lab attitude
        self.R = None
        
        # Lab-to-conv attitude
        self.convR = np.eye(4)
        
        # Frames for voxel array reconstruction
        self.voxFrames = 'auto'
        
        # Voxel array parameters
        self.xmin = None
        self.ymin = None
        self.zmin = None
        self.xmax = None
        self.ymax = None
        self.zmax = None
        self.xl = None
        self.yl = None
        self.zl = None
        self.xo = None
        self.yo = None
        self.zo = None
        self.fx = 1.
        self.fy = 1.
        self.fz = 1.
        
        # US images alignment parameters
        self.wrapper = 'none'
        self.step = 1
        self.alignFrames = None
        self.validKineFrames = None
        self.fillVoxMethod = 'avg'
        
        # Voxel array data
        self.V = None
        self.contV = None
        self.usedV = None
        self.internalV = None
        
        # vtkImageData properties
        self.sx = None
        self.sy = None
        self.sz = None
        
        # Gaps filling parameters
        self.method = 'none'
        self.blocksN = 100
        self.blockDir = 'X'
        self.maxS = 3
        self.distTh = None
        self.minPct = 0.
        
        # Features extracted from US images
        self.features = None
        
        # Precisions container
        self.acc = {}
        
        # Precisions container
        self.prec = {}
        
        # Time delay between devices
        self.timeDelay = 0.
        
        
    
    def setKineFiles(self, kineFiles):
        """Set kinematics files list.
        
        Parameters
        ----------
        kineFiles : list
            List of kinematics files.
        """
        
        checkKineFiles(kineFiles)
        self.kineFiles = kineFiles
        
    
    def getKineFiles(self):
        """Get kinematics files list.
        
        Returns
        -------
        list
            List of kinematics files.
        """
        
        return self.kineFiles
    

    def setUSFiles(self, usFiles):
        """Set US files list.
        
        Parameters
        ----------
        usFiles : list
            List of US files.
        """
        
        checkUsFiles(usFiles)
        self.usFiles = usFiles
        
    
    def getUSFiles(self):
        """Get US files list.
        
        Returns
        -------
        list
            List of US files.
        """
        
        return self.usFiles
        
        
    def setDataSourceProperties(self, **kwargs):
        """Set data source properties (for US and/or optoelectronic system).
        
        Parameters
        ----------
        kineFreq : int, optional
            Optoelectronic system frequency (in *Hz*).
        
        USFreq : int, optional
            US system frequency (in *Hz*).
        
        w : int, optional
            US image width (in *pixels*).
        
        h : int, optional
            US image height (in *pixels*).
        
        USTimeVector : list, optional
            List of time instants (in *s*) in which US frame were recorded.
            If multiple US files are provided in ``setUSFiles()``, the length 
            of this parameter must be equal to the sum of the frame numbers for
            each US file.
        
        fromUSFiles : list, optional
            List DICOM file paths from which to extract US data properties.
            If specified, w, h, USTimeVector, USFreq provided as input will be 
            ignored and will be parsed from these files
        
        pixel2mmX, pixel2mmY : float; optional
            Number of mm for each pixel in US image, for horizontal and 
            vertical axis (in *mm/pixel*).
        
        """
        
        # Check kineFreq
        if 'kineFreq' in kwargs:
            kineFreq = kwargs['kineFreq']
            checkFreq(kineFreq)
            self.kineFreq = kineFreq
        
        # Check reading method
        if 'fromUSFiles' in kwargs:
            
            # Read US files
            filePaths = kwargs['fromUSFiles']
            checkUsFiles(filePaths)
            usTimeVector = []
            usFreqPrev = None
            for i in xrange(0, len(filePaths)):
                filePath = filePaths[i]
                print 'Getting US image properties from file {0}...'.format(filePath)
                checkFilePath(filePath)
                D, ds = readDICOM(filePath)
                
                # Get w
                w = ds.Columns
                checkImDim(w)
                if i == 0:
                    wPrev = w
                else:
                    if w <> wPrev:
                        raise Exception('w changes across files')
                
                # Get h
                h = ds.Rows
                checkImDim(h)
                if i == 0:
                    hPrev = h
                else:
                    if h <> hPrev:
                        raise Exception('h changes across files')
                
                # Get USFreq
                if 'FrameTimeVector' not in ds:
                    if 'CineRate' not in ds:
                        raise Exception('If FrameTimeVector is not found, CineRate must be present')
                    usFreq = float(ds.CineRate)
                    checkFreq(usFreq)
                    if i == 0:
                        usFreqPrev = usFreq
                    else:
                        if usFreq <> usFreqPrev:
                            raise Exception('USFreq changes across files')
                
                # Get USTimeVector
                if 'FrameTimeVector' in ds:
                    usTimeVectorTemp = ds.FrameTimeVector
                    checkTimeVector(usTimeVectorTemp)
                    print 'FrameTimeVector found in file {0}'.format(filePath)
                    usTimeVector.append((np.cumsum(usTimeVectorTemp) / 1000).tolist())
                else:
                    if 'NumberOfFrames' in ds:
                        N = int(ds.NumberOfFrames)
                    else:
                        N = D.shape[1]
                    usFreq = float(ds.CineRate)
                    usTimeVectorTemp = (np.arange(0, N) / usFreq).tolist()
                    print 'FrameTimeVector not found in file {0}. It will be generate by using CineRate and the number of frames.'.format(filePath)
                    usTimeVector.append(usTimeVectorTemp)
                
            
                del D, ds
            
            # Get properties
            self.w = wPrev
            self.h = hPrev
            self.usFreq = usFreqPrev
            self.usTimeVector = usTimeVector
            print 'US image properties got from files'
        
        else:
        
            # Check w
            if 'w' in kwargs:
                w = kwargs['w']
                checkImDim(w)
                self.w = w
                    
            # Check h
            if 'h' in kwargs:
                h = kwargs['h']
                checkImDim(h)
                self.h = h
            
            # Check USFreq
            if 'USFreq' in kwargs:
                usFreq = kwargs['USFreq']
                checkFreq(usFreq)
                self.usFreq = usFreq
            
            # Check USTimeVector
            if 'USTimeVector' in kwargs:
                usTimeVector = kwargs['USTimeVector']
                for item in usTimeVector:
                    checkTimeVector(item)
                self.usTimeVector = usTimeVector
        
        
        # Check pixel2mmX
        if 'pixel2mmX' in kwargs:
            pixel2mmX = kwargs['pixel2mmX']
            checkPixel2mm(pixel2mmX)
            self.pixel2mmX = pixel2mmX

        # Check pixel2mmY
        if 'pixel2mmY' in kwargs:
            pixel2mmY = kwargs['pixel2mmY']
            checkPixel2mm(pixel2mmY)
            self.pixel2mmY = pixel2mmY
            
        # Adjust USTimeVector
        self.adjustUSTimeVector()
         
    
    def calculatePoseForUSProbe(self, mkrList=['M1','M2','M3','M4'], USProbePoseFun='default', USProbePoseFunArgs=None, globPoseFun=None, globPoseFunArgs=None, kineFilesReadOpts={}, showMarkers=False):
        """Calculate the attitude (or pose) of the marker-based US probe reference frame with respect to the global reference frame.

        .. note::
        
            In [Ref2]_, this is named :math:`^{T}T_{R}`.
        
        After extracting markers data from the kinematics files set with method ``setKineFiles()``, this data will
        be concatenated and resampled using ``USTimeVector``, if this one is available. Otherwise, kinematics data
        will be resampled based on optoelectronic system frequency and US system frequency. Only after kinematics and
        US data have a common time line, US probe attitude will be calculated.
        
        Parameters
        ----------
        mkrList : list
            List of marker names to be extracted from kinematics files. These will be used for
            creating the probe reference frame, and the global reference frame, if requested.
        
        USProbePoseFun : mixed
            Function defining the US probe reference frame. 
            If function, it takes two input parameters. The first one is ``mkrList``, while the second one being
            a dictionary where the keys are marker names defined in ``mkrList``, and value are N x 3 Numpy arrays of 
            3D coordinates (N is the number of time frames). The function must return a 2-elements list; the first one 
            is a Numpy N x 3 x 3 rotation matrix from probe reference frame to laboratory reference frame; the second
            one is a N x 3 Numpy array representing 3D coordinates of US probe reference frame origin in laboratory 
            reference frame. 
            If string, it must be ``'default'``. In this case, the reference frame is defined as in the function 
            ``kine.markersClusterFun()``.
        
        USProbePoseFunArgs : mixed
            Additional parameters passed to ``USProbePoseFun``.
        
        globPoseFunArgs : mixed
            Additional parameters passed to ``globPoseFunArgs``.
        
        globPoseFun : mixed
            Function defining the global reference frame.
            This function takes the same input arguments as the function ``USProbePoseFun``. It must return a 2-elements
            list; the first one is a Numpy N x 3 x 3 rotation matrix from global reference frame to laboratory reference 
            frame; the second one is a N x 3 Numpy array representing 3D coordinates of the global reference frame origin
            in laboratory reference frame. 
            
        kineFilesReadOpts : dict
            Options for kinematics files reading. See parameter ``opts`` for ``kine.readC3D()`` function.
        
        showMarkers : bool
            If ``True``, show the marker data after resampling to US time line.    
        
        """
        
        # Check input validity
        checkMkrList(mkrList)
        checkKineFiles(self.kineFiles)
        
        # Read kinematic files
        fileNames = self.kineFiles
        mkrs = {}
        for m in mkrList:
            mkrs[m] = np.zeros((0,3))
        Nf = [0] * len(fileNames)
        timeLine = np.empty((0,))
        self.validKineFrames = np.empty((0,), dtype=np.int32)
        for i in xrange(0, len(fileNames)):
            fileName = fileNames[i]
            print 'Reading C3D file {0} ...'.format(fileName)
            markersData = readC3D(fileName, ['markers'], opts=kineFilesReadOpts)['markers']
            Nf[i] = markersData[mkrList[0]].shape[0]
            newMkrs = {}
            for m in mkrList:
                newMkrs[m] = markersData[m]
            print '{0} frames found'.format(Nf[i])
            
            # Resample markers data
            print 'Resampling markers data to US time frame...'
            if self.usTimeVector <> None:
                if len(self.usTimeVector) <> len(self.kineFiles):
                    raise Exception('Number of kinematic files must be the same as the number of US time vectors')
                checkTimeVector(self.usTimeVector[i])
                checkFreq(self.kineFreq)
                print 'Kinematics data resampling will be based on US data time vector'
                resampleStep = None
                currTimeVector = self.usTimeVector[i]
                if None in currTimeVector:
                    currTimeVector = np.linspace(0, (Nf[i]-1)*dt, num=len(currTimeVector)) 
            else:
                raise Exception('Impossible to resample kinematics data')
            currUSTimeVector = np.array(self.usTimeVectorAdj[i])
            
            newMkrsResampled = {}
            for m in mkrList:
                newMkrsResampled[m], xInterpInd = resampleMarker(newMkrs[m], step=resampleStep, x=currUSTimeVector, origFreq=self.kineFreq)
            print 'Markers data resampled'
            
            # Get frame numbers for not extrapolated kine frames 
            if i == 0:
                Nprev = 0
            else:
                Nprev = np.sum(Nf[:i])
            self.validKineFrames = np.append(self.validKineFrames, Nprev + xInterpInd)
            
            # Append marker data
            for m in mkrList:
                mkrs[m] = np.vstack((mkrs[m], newMkrsResampled[m]))
                
            print 'C3D file read'
        
        # Resampling markers data to US frequency
        print 'Frames number before resampling: {0}'.format(sum(Nf))
        print 'Frames number after resampling: {0}'.format(mkrs[mkrList[0]].shape[0])

        # Show markers on request        
        if showMarkers == True:
            
            C = ['X', 'Y', 'Z']
            
            # Loop for each marker
            for m in xrange(len(mkrList)):
                
                mkrName = mkrList[m]
                mkrData = mkrs[mkrName][self.validKineFrames,:]
                
                # Loop for each coordinate 
                for c in xrange(len(C)):
                    
                    p = m * len(C) + c + 1
                    plt.subplot(len(mkrList), len(C), p)
                    timeLine = np.arange(mkrData.shape[0])
                    plt.plot(timeLine, mkrData[:,c])
                    plt.title('%s (%s)' % (mkrName, C[c]))
                    plt.ylabel('[mm]')
                    if m == len(mkrList) - 1:
                        plt.xlabel('Frames')
                    
            # Show data
            plt.subplots_adjust(wspace=0.5, hspace=0.8)
            plt.show()

        # Calculate affine matrix from probe to laboratory reference frame
        print 'Calculating US probe roto-translation matrix for all time frames ...'
        if USProbePoseFun == 'default':
            _USProbePoseFun = markersClusterFun
        else:
            _USProbePoseFun = USProbePoseFun
        if USProbePoseFunArgs is None or USProbePoseFun == 'default':
            Rpr, Tpr = _USProbePoseFun(mkrs, mkrList)
        else:
            Rpr, Tpr = _USProbePoseFun(mkrs, mkrList, USProbePoseFunArgs)
        RprFull = composeRotoTranslMatrix(Rpr, Tpr)
               
        if globPoseFun <> None:
            
            # Calculate affine matrix from global reference frame to laboratory reference frame
            if globPoseFunArgs is None:
                Rgl, Tgl = globPoseFun(mkrs, mkrList)
            else:
                Rgl, Tgl = globPoseFun(mkrs, mkrList, globPoseFunArgs)
            RglFull = composeRotoTranslMatrix(Rgl, Tgl)
            
            # Calculate affine matrix from probe reference frame to global reference frame
            glRprFull = dot3(np.linalg.inv(RglFull), RprFull)
            
        else:
            
            glRprFull = RprFull
        
        # Get back rotation matrix and translation
        self.Rpr, self.Tpr = decomposeRotoTranslMatrix(glRprFull)
        
        print 'US probe roto-translation matrix calculated'
        
        
        
    def calculatePoseForUSImages(self):
        """Calculate the pose of the US images with respect to the global reference frame.
        
        .. note::
        
            In [Ref2]_, this is the product :math:`^{T}T_{R}\ ^{R}T_{P}`.
            
        """
        
        # Check input validity
        checkIm2PrPose(self.prRim, self.Tim)
        checkPr2GlPose(self.Rpr, self.Tpr)
        
        print 'Calculating US images roto-translation matrix for all time frames ...'        
        
        # Calculate rotation matrix for pixel to world
        R = np.dot(self.Rpr, self.prRim) # N x 3 x 3
        T = np.dot(self.Rpr, self.Tim) + self.Tpr
        
        # Create affine transformation matrix (N x 4 x 4)
        Rfull = composeRotoTranslMatrix(R, T)
        
        print 'US images roto-translation matrix calculated'
        
        self.R = Rfull
        
        
    def getPoseForUSImages(self):
        """Get the pose of the US images with respect to the global reference frame.
        
        Returns
        -------
        np.ndarray
            N x 4 x 4 pose, for N time frames.
            
        """  
        
        # Check input validity
        checkIm2GlPose(self.R)
        
        return self.R
        
        
    def getImageCornersAs3DPoints(self):
        """Create virtual 3D points for US images corners with respect to the global reference frame.
        
        Returns
        -------
        dict
            Dictionary where keys are 4 marker names and values are np.ndarray
            N x 3 matrices, representing point coordinates, for N time frames.
            The following are the points created:
            
            - im_TR: top-right corner
            - im_BR: bottom-right corner
            - im_TL: top-left corner
            - im_BL: bottom-left corner
            
            
        """  
        
        # Check input validity
        checkIm2GlPose(self.R)
        checkImDim(self.w)
        checkImDim(self.h)
        checkPixel2mm(self.pixel2mmX)
        checkPixel2mm(self.pixel2mmY)    
    
        # Create virtual points for corners
        pc = createImageCorners(self.w, self.h, self.pixel2mmX, self.pixel2mmY)
        pcg = np.dot(self.R,pc)[:,0:3,:]    # N x 3 x 4
        points = {}
        points['im_TR'] = pcg[:,:,0]
        points['im_BR'] = pcg[:,:,1]
        points['im_TL'] = pcg[:,:,2]
        points['im_BL'] = pcg[:,:,3]
        return points
        
    
    def extractFeatureFromUSImages(self, feature='2_points_on_line', segmentation='manual', segParams={}, showViewer=True, featuresFile=None):
        """Extract features (points, lines, ...) from US images.
        
        The used file will be the one indicated in method ``setUSFiles()``.
        
        Parameters
        ----------
        feature : str
            target feature type.
            If '2_points_on_line', the features under consideration are 2 points on the longest edge line in the image.
            If '2_points', the features under consideration are 2 manually defined points.
            If '1_point', the feature under consideration is 1 manually defined point.
            If 'mask', the feature under consideration is a suset of pixels of the image.
        
        segmentation : str
            Segmentation method.
            If 'manual', an interactive window will be popped up and the user will be able to select manually the features image per image. 
            If 'auto_hough' (only for ``feature='2_points_on_line'``), the longest line will be automatically detected by the Hough transform and 2 points will be place on that line according to ``segParams``.
        
        segParams : dict
            Parameters for features extraction.
            If ``segmentation='auto_hough'``:
            
            - 'par_seg': see ``parSeg`` in ``SegmentPointsHoughUI.__init__()``.
            - 'data_constr': see ``dataConstr`` in ``SegmentPointsHoughUI.__init__()``.
            - 'save_data_path': see ``saveDataPath`` in ``SegmentPointsHoughUI.__init__()``.
            
            If ``segmentation='mask'``, see ``maskParams`` in ``MaskImageUI.__init__()``.
        
        showViewer : bool
            If True, it pops up a viewer to show or edit the features.       
        
        featuresFile : mixed
            Features file path.
            If None, it will be ignored. Othwerwise, it must indicate the full path of a previously saved. This contains features data.
        
        """                
        
        # Check input validity
        checkFeature(feature)
        checkSegmentation(segmentation)
        if featuresFile <> None:
            checkFilePath(featuresFile)
        checkUsFiles(self.usFiles, L=1)
        
        # Load image if necessary
        #if featuresFile == None:
        # Read DICOM file
        print 'Reading DICOM file {0} ...'.format(self.usFiles[0])
        D, ds = readDICOM(self.usFiles[0])
        I = pixelData2grey(D)   # supposing D is "small" and fits in memory
        print 'Number of frames: {0}'.format(I.shape[0])
        print 'DICOM file read'
        
        # Load features from file is requested
        data = {}
        if featuresFile <> None:
            print 'Loading features file...'
            data = readFeaturesFile(featuresFile)
           
        # Perform feature extraction
        print 'Extracting features...'
        if feature == '2_points_on_line':
            title = 'Click on 2 points on the border line'
            if segmentation == 'manual':
                ui = SegmentPointsUI(2, data, I, title=title)
            if segmentation == 'auto_hough':
                parSeg = segParams['par_seg']
                dataConstr = segParams['data_constr']
                saveDataPath = segParams['save_data_path']
                ui = SegmentPointsHoughUI(2, parSeg, dataConstr, data, I, title=title, saveDataPath=saveDataPath)
        if feature == '2_points':
            title = 'Click on 2 points'
            if segmentation == 'manual':
                ui = SegmentPointsUI(2, data, I, title=title)
        if feature == '1_point':
            title = 'Click on 1 point'
            if segmentation == 'manual':
                ui = SegmentPointsUI(1, data, I, title=title)
        if feature == 'mask':
            title = 'Select a mask'
            ui = MaskImageUI(segParams, data, I, title=title)
        # Show viewer if requested
        if showViewer:
            ui.showViewer()
        ui.closeViewer()
        
        # Get features
        self.features = ui.getData()
            
        print 'Features extracted'
    

    def calibrateProbe(self, init, xtol=None, ftol=None, method='eq_based', method_args={'phantom':'single_wall','regularize_J':True}, fixed=[], correctResults=False):
        """Calculate the attitude (or pose) of the US images with respect to the probe reference frame.        
        
        .. note::
        
            In [Ref2]_, this is named :math:`^{R}T_{P}`.
        
        Parameters
        ----------
        init : dict
            Dictionary containing initial values for the calibration algorithm (see [Ref2]_, Table 1).
            Keys must belong to this list:
            
            - sx, sy: number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*)
            - x1, y1, z1: coordinates (in *mm*) of vector pointing from US probe reference frame origin to the US image reference frame origin.
            - gamma1, beta1, alpha1: rotation angles (in *rad*) representing consecutive rotations around the US image reference frame axis (X, Y and Z). This rotations would get it oriented as the US probe reference frame (see `here <http://kwon3d.com/theory/euler/euler_angles.html>`_ for more details). Use opposite sign with respect to the `right-hand rule <http://en.wikipedia.org/wiki/Right-hand_rule>`_
            - x2, y2, z2: coordinates (in *mm*) of vector pointing from global reference frame origin to the calibration phantom reference frame origin.
            - gamma2, beta2, alpha2: same meaning as gamma1, beta1, alpha1, but now the rotations are from global reference frame to probe reference frame.
    
            If ``method='eq_based'``, only the following variables have to be present: 'sx', 'sy', 'x1', 'y1', 'z1', 'alpha1', 'beta1', 'gamma1', 'x2', 'y2', 'z2', 'alpha2', 'beta2', 'gamma2'.
            If ``method='maximize_NCCint'`` or ``'maximize_NCC'``, only the following variables have to be present: 'x1', 'y1', 'z1', 'alpha1', 'beta1', 'gamma1'.

        xtol : float
            Relative error desired in the approximate solution (see argument ``options['xtol']`` or ``tol`` in ``scipy.optimize.root()``).
        
        ftol : float
            Relative error desired in the sum of squares (see argument ``options['ftol']`` in ``scipy.optimize.root()``).

        method : str
            Method used to estimate calibration parameters.
            If 'eq_based', a system of equations (representing contraints) will be solved (see [Ref2]_).
            If 'maximize_NCCint', the algorithm used is a modification of the one described in [Ref3]_. It aims at maximizing the average Normalized Cross-Correlation of the intersection of pair of US images.
            If 'maximize_NCC' or 'maximize_NCCfast', the algorithm used is described in [Ref3]_.   
        
        method_args : dict
            Further arguments for method used.
            If ``method='eq_based'``, it must contain the following keys:
            
            - 'phantom': calibration phantom type (see [Ref2]_). 
              If 'single_wall', the calibration equations system is solved by using formula 8 in [Ref2]_. Variables x2, y2, alpha2 will be forced to 0.
            
            If ``method='maximize_NCCint'``, it must contain the following keys:
            
            - 'frames': If 'all_combos_in', then all the frames combinations in a interval will be used. If list, each element must be a list of 2 elements, representing a frames combination for NCC calculation.
            - 'frames_interval': see 'frames'.
            
            NCC values, each one related to a couple of frames, will be averaged.
    
            If ``method='maximize_NCC'`` or ``method='maximize_NCCfast'``, it must contain the following keys:
            
            - 'sweep_frames': 2-elem list where the first element is a list of original images sweep frames and the second element is a 2-elem list defining start and end frame of the reconstruction sweep. 
            - 'imag_comp_save_path': if not empty, it will be used to save each the couple original image - reconstruction for each iteration. 
              Each file name is in the format it<itn>_im<ofn>.jpeg, where <itn> is the iteration number (for Nelder-Mead method), <ofn> is the original image frame number.
            - 'max_expr': expression to maximize.
              If 'avg_NCC', the NCCs calculated for each wanted pair original frame vs reconstruction template will be averaged.
              If 'weighted_avg_NCC', the NCCs calculated for each wanted pair original frame vs reconstruction template will be averaged using as weigths the percentage of reconstructed template.
              This percentage, in the bottom-left picture in the figures saved in 'imag_comp_save_path', corresponds to the ratio between the area occupied by the straight lines and the image size.
            
            Common parameters for all NCC-based methods:
    
            - 'th_z': threshold value (in *mm*) under which points on a reconstruction sweep can be considered belonging to an original image plane.
            
            NCC values, each one related to one frame from the first sweep and the reconstruction sweep, will be averaged.
        
        fixed : list
            List of variable name for which the value is exactly known.
            These variables become constant in the calibration equations. For the list of allowed names, see argument ``init``.
        
        correctResults : bool
            Correct for mirror solutions.
            According to the Appendix of [Ref2]_, calculated variables could bring to 'mirror solutions'. This flag will bring them to a standard form.
            
        """
        
        # Check input validity 
        checkCalibMethod(method)
        if method == 'eq_based':
            phantom = method_args['phantom']
            checkPhantom(phantom)
            checkFeatures(self.features)
        if method in ['maximize_NCCint', 'maximize_NCC', 'maximize_NCCfast']:
            checkPixel2mm(self.pixel2mmX)
            checkPixel2mm(self.pixel2mmY)
            checkUsFiles(self.usFiles, L=1)
        checkPr2GlPose(self.Rpr, self.Tpr)
        
        # Create expressions
        if method == 'eq_based':
            print 'Creating calibration equations...'
            eq, J, prTi, syms, allVariables, mus = createCalibEquations()
            variables = allVariables[:]
            print 'Equations defined'
                
            # Set to 0 some variables depending on phantom
            if phantom == 'single_wall':
                # Select equation
                eq = eq[2,0]    # select 3rd equation
                J = J[2,:]      # select 3d equation
                J.col_del(8)    # delete derivatives for x2
                J.col_del(8)    # delete derivatives for y2
                J.col_del(9)    # delete derivatives for alpha2
                # Set to 0 some variables
                eq = eq.subs([(syms['x2'],0),(syms['y2'],0),(syms['alpha2'],0)])
                J = J.subs([(syms['x2'],0),(syms['y2'],0),(syms['alpha2'],0)])
                del syms['x2'], syms['y2'], syms['alpha2']
                variables.remove('x2')
                variables.remove('y2')
                variables.remove('alpha2')
                # Delete unwanted variables
                for f in fixed:
                    J.col_del(variables.index(f))
                    eq = eq.subs([(syms[f],init[f])])
                    J = J.subs([(syms[f],init[f])])
                    del syms[f]
                    variables.remove(f)
        elif method in ['maximize_NCCint', 'maximize_NCC', 'maximize_NCCfast']:
            i2Ti1, prTi, syms, allVariables, mus = createCalibExpressionsForMaxNCC()
            variables = allVariables[:]
            # Delete unwanted variables
            for f in fixed:
                i2Ti1 = i2Ti1.subs([(syms[f],init[f])])
                del syms[f]
                variables.remove(f)
            
        # Check variables init values
        if set(set(variables)).issubset(init.keys()) == False:
            raise Exception('Some variables were not initialized')
        initValues = [init[variables[i]] for i in xrange(0,len(variables))]
                
        # Solve the equations
        print 'List of variables: {0}'.format(variables)
        print 'List of initial values: {0}'.format(initValues)
        print 'Solving calibration...'
        if method == 'eq_based':
            regJ = frames = method_args['regularize_J']
            sol, kond = solveCalibEquations(eq, J, syms, variables, initValues, xtol, ftol, self.Rpr, self.Tpr, self.features, regJ)
        elif method in ['maximize_NCCint','maximize_NCC','maximize_NCCfast']:
            # Read DICOM file
            D, ds = readDICOM(self.usFiles[0])
            I = pixelData2grey(D)
            if method == 'maximize_NCCint':
                # Create frames couples
                frames = method_args['frames']
                if frames == 'all_combos_in':
                    framesInt = method_args['frames_interval']
                    comb = combinations(tuple(framesInt), 2)
                    frames = [co for co in comb]
                # Run calibration
                sol = maximizeNCCint(i2Ti1, syms, variables, initValues, self.Rpr, self.Tpr, I, self.pixel2mmX, self.pixel2mmY, frames)
                kond = 0
            elif method in ['maximize_NCC', 'maximize_NCCfast']:
                # Get options
                frames = method_args['sweep_frames']
                path = method_args['imag_comp_save_path']
                thZ = method_args['th_z']
                maxExpr = method_args['max_expr']
                # Get mask, if existing
                if self.features == None:
                    mask = None
                else:
                    No = len(frames[0])
                    mask = np.zeros((No, I.shape[1], I.shape[2]), dtype=np.bool)
                    for i in xrange(No):
                        mask[i,:,:] = self.features[frames[0][i]]
                # Run calibration
                if method == 'maximize_NCC':
                    NCCfunction = maximizeNCC
                else:
                    NCCfunction = maximizeNCCcy
                sol = NCCfunction(i2Ti1, syms, variables, initValues, self.Rpr, self.Tpr, I, self.pixel2mmX, self.pixel2mmY, frames, path, thZ, maxExpr, mask=mask)
                kond = 0                
        print 'Iterations terminated ({0})'.format(sol.message)
        
        if sol.success:
            # Show conditioning number
            print 'Condition number: %d' % kond
            # Create solution dictionary
            print 'Calibration succesfully solved' 
            x = {}
            for v in allVariables:
                if v in variables:
                    x[v] = sol.x[variables.index(v)]
                else:
                    if v in init:
                        x[v] = init[v]
                    else:
                        x[v] = 0.
            # Correct results if wanted
            if correctResults:
                if method == 'eq_based':
                    print 'Correcting results...'
                    for a in ['alpha1', 'beta1', 'gamma1', 'alpha2', 'beta2', 'gamma2']:
                        x[a] = setInsideRange(x[a], np.pi, 2*np.pi)
                    val = setInsideRange(x['beta1'], np.pi/2, np.pi)
                    if val <> x['beta1']:
                        x['beta1'] = val
                        x['alpha1'] += np.pi
                        x['gamma1'] += np.pi
                    if x['sy'] < 0:
                        x['gamma1'] += np.pi
                        x['sy'] = -x['sy']
                    if x['sx'] < 0:
                        x['alpha1'] += np.pi
                        x['beta1'] = -x['beta1']
                        x['gamma1'] = np.pi - x['gamma1']
                        x['sx'] = -x['sx']
                    for a in ['alpha1', 'gamma1', 'alpha2', 'gamma2']:
                        x[a] = setInsideRange(x[a], np.pi, 2*np.pi)
                    print 'Results corrected'
            # Calculate image-to-probe attitude
            subs = {}
            subs['x1'] = x['x1']
            subs['y1'] = x['y1']
            subs['z1'] = x['z1']
            subs['alpha1'] = x['alpha1']
            subs['beta1'] = x['beta1']
            subs['gamma1'] = x['gamma1']
            prTi = prTi.evalf(subs=subs)
            prTi = np.array(prTi).astype(np.float)
            prRim = prTi[0:3,0:3]
            Tim = prTi[0:3,3].squeeze()
            # Extract pixem2mm values
            if 'sx' in x:
                sx = x['sx']
            else:
                sx = self.pixel2mmX
            if 'sy' in x:
                sy = x['sy']
            else:
                sy = self.pixel2mmY
            # Print results 
            for v, mu in zip(allVariables, mus):
                if mu == 'rad':
                    val = np.rad2deg(x[v])
                    mu = 'deg'
                else:
                    val = x[v]
                print v + (': %f ' % val) + mu            
        else:
            raise Exception('System not succesfully solved' )

        # Set data internally
        self.prRim = prRim
        self.Tim = Tim
        self.pixel2mmX = sx
        self.pixel2mmY = sy
        self.calib = {}
        self.calib['root_sol'] = sol
        self.calib['root_vars'] = variables
        if method == 'eq_based':
            self.calib['RMS'] = np.sqrt(np.mean(sol.fun**2))
        else:
            self.calib['RMS'] = 0.
        self.calib['kond'] = kond
        if method == 'eq_based':
            self.calib['cov_x'] = sol.cov_x * (sol.fun**2).sum() / (sol.fun.shape[0] - len(variables))
        else:
            self.calib['cov_x'] = 0.
    
    
    def getProbeCalibrationData(self):
        """Get calibration results.
        
        Returns
        -------
        prRim : np.ndarray
            3 x 3 rotation matrix from US image reference frame to probe reference frame.
        
        Tim : np.ndarray
            3-elem vector (in *mm*), expressed in probe reference frame, from probe reference frame origin to US image reference frame origin.
        
        sx, sy : float
            See method ``calibrateProbe()``.
        
        calib : dict
            Dictionary with the following fields:
        
            - root_sol (*Result*) – contains output from ``scipy.optimize.root()``.
            - root_vars (*list*) – contains variables names (see argument ``init`` for argument ``calibrateProbe()``).
            - RMS (*float*) – RMS of the equations residuals (only for ``method='eq_based'`` in ``calibrateProbe()``)
            - kond (*int*) – Condition number calculated as the ratio between max and min eigenvalues from the SVD decomposition of *Jacobian* matrix calculated in the solution point (only for ``method='eq_based'`` in ``calibrateProbe()``).
            - cov_x (*np.ndarray*) – covariance matrix differing from calib['root_sol'].cov_x for a multiplying factor being sum of squared residuals divided by degrees of freedom (only for ``method='eq_based'`` in ``calibrateProbe()``).
        
        """

        return self.prRim, self.Tim, self.pixel2mmX, self.pixel2mmY, self.calib


    def setProbeCalibrationData(self, prRim, Tim):
        """Set probe calibration data.
        
        Parameters
        ----------
        prRim : np.ndarray
            See method ``getProbeCalibrationData()``.
        Tim : np.ndarray
            See method ``getProbeCalibrationData()``.
            
        """
        
        # Check input validity
        checkIm2PrPose(prRim, Tim)
            
        self.prRim, self.Tim = prRim, Tim
        
    
    def evalCalibMatrix(self, x):
        """*(static)* Evaluate calibration matrix with parameters values.
        
        Parameters
        ----------
        x : dict
            See param ``init`` for function ``calibrateProbe()``. Only the following keys will be used: alpha1, beta1, gamma1, x1, y1, z1.
        
        Returns
        -------
        prRim : np.ndarray 
            3 x 3 rotation matrix from US image reference frame to probe reference frame.
        
        Tim : np.ndarray
            3-elem vector (in *mm*), expressed in probe reference frame, from probe reference frame origin to US image reference frame origin.
        
        """
        
        # Get calibration matrix expression
        prTi, syms = creatCalibMatrix()
        
        # Evaluate the expression
        subs = {}
        subs['x1'] = x['x1']
        subs['y1'] = x['y1']
        subs['z1'] = x['z1']
        subs['alpha1'] = x['alpha1']
        subs['beta1'] = x['beta1']
        subs['gamma1'] = x['gamma1']
        prTi = prTi.evalf(subs=subs)
        prTi = np.array(prTi).astype(np.float)
        
        # Get attitude and translation
        prRim = prTi[0:3,0:3]
        Tim = prTi[0:3,3].squeeze()
        return prRim, Tim
        
        
        
    def setValidFramesForVoxelArray(self, voxFrames='auto', voxFramesBounds=None):
        """Set the list of frames (US time line) of the images that can be contained in the voxel array.
        Frames are further filtered out based on the invalid kinematics frames calculated 
        by ``calculatePoseForUSProbe()``.
        
        Parameters
        ----------
        voxFrames : mixed
            List of US time frames.
            If 'auto', all the frames without missing optoelectronic data information will be considered.       
            If 'all', all the frames will be considered.
            If list, it must contain the list of frames to be considered.
        
        voxFramesBounds : mixed
            Bounding frames for the list of frames to be contained in the voxel array.
            If None, all the frames out of ``voxFrames`` will be used.
            If list, it must contain 2 elements specifying lower and upper bround frames for the list in ``voxFrames``.
            
        """
        
        # Check input validity
        checkIm2GlPose(self.R)
        checkVoxFrames(voxFrames, self.R.shape[0])
        checkVoxFramesBounds(voxFramesBounds, self.R.shape[0])
        
        # Create voxel frames indices 
        if voxFrames == 'all':
            voxFrames = range(0, self.R.shape[0])
        elif voxFrames == 'auto':
            voxFrames = (np.delete(np.arange(self.R.shape[0]), np.nonzero(np.isnan(self.R))[0])).tolist()
        
        # Creae voxel frames bounds if not existing
        if voxFramesBounds is None:
            voxFramesBounds = [0, self.R.shape[0]-1]
        
        # Limit voxel frames to bounds
        voxFrames = np.array(voxFrames)
        voxFrames = voxFrames[(voxFrames >= voxFramesBounds[0]) & (voxFrames <= voxFramesBounds[1])]
        
        # Intersect with the valid kinematics frames
        voxFrames = np.intersect1d(voxFrames, self.validKineFrames).tolist()
        self.voxFrames = voxFrames
        
        
    def calculateConvPose(self, convR):
        """Calculate roto-translation matrix from global reference frame to *convenient* reference frame.
        Voxel-array dimensions are calculated in this new refence frame. This rotation is important whenever the US scans sihouette is remarkably
        oblique to some axis of the global reference frame. In this case, the voxel-array dimensions (calculated by the smallest parallelepipedon 
        wrapping all the realigned scans), calculated in the global refrence frame, would not be optimal, i.e. larger than necessary.
        
        .. image:: diag_scan_direction.png
           :scale: 30 %          
          
        Parameters
        ----------
        convR : mixed
            Roto-translation matrix.
            If str, it specifies the method for automatically calculate the matrix.
            If 'auto_PCA', PCA is performed on all US image corners. The x, y and z of the new convenient reference frame are represented by the eigenvectors out of the PCA.
            If 'first_last_frames_centroid', the convenent reference frame is expressed as:
            
            - x from first image centroid to last image centroid
            - z orthogonal to x and the axis and the vector joining the top-left corner to the top-right corner of the first image
            - y orthogonal to z and x
            
            If np.ndarray, it must be manually specified as a 4 x 4 affine matrix.
            
        """
        
        # Check input validity
        checkIm2GlPose(self.R)
        checkVoxFrames(self.voxFrames, self.R.shape[0])
        checkGl2ConvPose(convR)
        self.convR = convR
        checkImDim(self.w)
        checkImDim(self.h)
        checkPixel2mm(self.pixel2mmX)
        checkPixel2mm(self.pixel2mmY)    
    
        # Calculating best pose automatically, if necessary
        ivx = np.array(self.voxFrames)
        pc = createImageCorners(self.w, self.h, self.pixel2mmX, self.pixel2mmY)
        if self.convR == 'auto_PCA':
            # Perform PCA on image corners
            print 'Performing PCA on images corners...'
            pcg = np.dot(self.R[ivx,:,:],pc)[:,0:3,:]    # N x 3 x 4
            pcg = np.reshape(pcg.transpose(1,2,0), (3,4*ivx.shape[0]), order='F')
            U, s = pca(pcg)
            # Build convenience affine matrix
            self.convR = np.vstack((np.hstack((U,np.zeros((3,1)))),[0,0,0,1])).T
            print 'PCA perfomed'
        elif self.convR == 'first_last_frames_centroid':
            # Search connection from first image centroid to last image centroid (X)
            print 'Performing convenient reference frame calculation based on first and last image centroids...'
            pcg = np.dot(self.R[ivx,:,:],pc)[:,0:3,:]   # N x 3 x 4
            C0 = np.mean(pcg[ivx.min(),:,:], axis=1)  # 3
            C1 = np.mean(pcg[ivx.max(),:,:], axis=1)  # 3
            X = C1 - C0
            # Define Y and Z axis
            corners0 = pcg[ivx.min(),:,:] # 3 x 4
            Ytemp = corners0[:,0] - corners0[:,2]   # from top-left corner to top-right corner
            Z = np.cross(X, Ytemp)
            Y = np.cross(Z, X)
            # Normalize axis length
            X = X / np.linalg.norm(X)
            Y = Y / np.linalg.norm(Y)
            Z = Z / np.linalg.norm(Z)
            # Create rotation matrix
            M = np.array([X, Y, Z]).T
            # Build convenience affine matrix
            self.convR = np.vstack((np.hstack((M,np.zeros((3,1)))),[0,0,0,1])).T
            print 'Convenient reference frame calculated'
            
    
    def getVoxelArrayPose(self):
        """Return roto-translation matrix from voxel array reference frame to global reference frame.

        Returns
        -------
        np.ndarray
            4 x 4 rototranslation matrix.
            
        """
        
        # Define roto-translation from convenient to global reference frame
        Tconv = np.linalg.inv(self.convR)
        
        # Define roto-translation from voxel-array to convenient reference frame
        convTva = np.eye(4)
        convTva[0:3,3] = [self.xmin,self.ymin,self.zmin]
        
        # Define roto-translation from voxel-array to global reference frame
        Tva = np.dot(Tconv, convTva)
        
        return Tva
    
    
    
    def setScaleFactors(self, fxyz, voxFramesBounds=None):
        """Set or calculate scale factors that multiply real voxel-array dimensions.
        
        Parameters
        ----------
        fxyz : mixed
            Scale factors.
            If list, it must contain 3 elements being the scale factors
            If 'auto_bounded_parallel_scans', the following should hold:
            
            - the US probe motion is supposed to be performed mainly along one axis (X);
            - corners of the US images during acquisition are supposed to not deviate too much from a straight line (along X);
            - motion velocity is supposed to be constant;
            - pixel/mm for US images are very similar for width and height.
            
            Scale factors are calculated as follows:
            
            - fx: ceil(abs((voxFramesBounds[1] - voxFramesBounds[0]) / (C1 - C0)));
            - fy, fz: ceil(1 / pixel2mmX).
            
            where:
            
            - C0 and C1 are the X coordinates (in *mm*) of the US image centers at frames ``voxFramesBounds[0]`` and ``voxFramesBounds[1]``;
            - pixel2mmX is the conversion factor (in *mm/pixel*) for width in the US images.
            
            See chapter :ref:`when-mem-error` for the use of these scale factors.
        
        voxFramesBounds : mixed
            Bounding frames for the list of frames to be contained in the voxel array.
            If None, first and last time frames out of ``setValidFramesForVoxelArray()`` will be used.
            If list, it must contain 2 elements specifying lower and upper bround frames.
        
        """
        
        # Check input validity
        checkFxyz(fxyz)
        checkIm2GlPose(self.R)
        checkVoxFrames(self.voxFrames, self.R.shape[0])
        checkVoxFramesBounds(voxFramesBounds, self.R.shape[0])
        checkGl2ConvPose(self.convR)
        checkImDim(self.w)
        checkImDim(self.h)
        checkPixel2mm(self.pixel2mmX)
        checkPixel2mm(self.pixel2mmY)
        
        # Creae voxel frames bounds if not existing
        if voxFramesBounds is None:
            voxFramesBounds = [self.voxFrames[0], self.voxFrames[-1]]
        
        # Calculating scale factors
        ivx = np.array(self.voxFrames)
        pc = createImageCorners(self.w, self.h, self.pixel2mmX, self.pixel2mmY)
        pcg = np.dot(dot2(self.convR,self.R[ivx,:,:]),pc)  # N x 4 x 4 (#frames x #coords+1 x #points)
        ivx = np.array(self.voxFrames)
        if fxyz == 'auto_bounded_parallel_scans':
            i0 = voxFramesBounds[0]
            i1 = voxFramesBounds[1]
            if i0 not in ivx:
                raise Exception('Frame %d is not a valid kinematic frame' % i0)
            if i1 not in ivx:
                raise Exception('Frame %d is not a valid kinematic frame' % i1)
            C0 = np.mean(pcg[ivx==i0,0:3,:].squeeze(), axis=1)[0]
            C1 = np.mean(pcg[ivx==i1,0:3,:].squeeze(), axis=1)[0]
            fx = np.ceil(np.abs((i1 - i0) / (C1 - C0)))
            fy = np.ceil(1. / self.pixel2mmX)
            fz = fy
        else:
            fx, fy, fz = fxyz[0], fxyz[1], fxyz[2]
        self.fx, self.fy, self.fz = fx, fy, fz
        print 'Scale factors fx, fy, fz set to: %d, %d, %d' % (self.fx, self.fy, self.fz)
        
    
    def calculateVoxelArrayDimensions(self):
        """Calculate dimensions for voxel array. The convenient reference frame
        (see ``calculateConvPose()``) is translated to a *voxel array* reference
        frame, optimally containing the US images is the first quadrant.
        """
        
        # Check input validity
        checkFxyz([self.fx, self.fy, self.fz])
        checkIm2GlPose(self.R)
        checkGl2ConvPose(self.convR)
        checkVoxFrames(self.voxFrames, self.R.shape[0])
        checkImDim(self.w)
        checkImDim(self.h)
        checkPixel2mm(self.pixel2mmX)
        checkPixel2mm(self.pixel2mmY)

        # Calculate coordinates for all points in convevient reference frame        
        pc = createImageCorners(self.w, self.h, self.pixel2mmX, self.pixel2mmY)
        ivx = np.intersect1d(np.array(self.voxFrames), self.validKineFrames)
        pcg = np.dot(dot2(self.convR,self.R[ivx,:,:]),pc)  # N x 4 x 4 (#frames x #coords+1 x #points)
        
        # Calculate volume dimensions
        print 'Calculating voxel array dimension ...'
        xmin, xmax = np.amin(pcg[:,0]), np.amax(pcg[:,0])
        ymin, ymax = np.amin(pcg[:,1]), np.amax(pcg[:,1])
        zmin, zmax = np.amin(pcg[:,2]), np.amax(pcg[:,2])
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax       
        self.zmin, self.zmax = zmin, zmax
         
        # Calculate voxel array size
        self.xl = (np.round(self.fx * xmax) - np.round(self.fx * xmin)) + 1
        self.yl = (np.round(self.fy * ymax) - np.round(self.fy * ymin)) + 1
        self.zl = (np.round(self.fz * zmax) - np.round(self.fz * zmin)) + 1
        self.xo = np.round(self.fx * xmin)
        self.yo = np.round(self.fy * ymin)
        self.zo = np.round(self.fz * zmin)
        print 'Voxel array dimension: {0} x {1} x {2}'.format(self.xl,self.yl,self.zl)
        
    
    def initVoxelArray(self):
        """Initialize voxel array. It instantiate data for the voxel array grey values.
        """
    
        
        # Create voxel array for grey values
        
        # NOTE:
        #
        # For a flat structure:
        # 
        # pros:
        # - it is already usable for a VTK data copy by numpy_to_vtk, without any ravel (MemoryError avoided)
        #
        # cons:
        # - direct block slicing takes some overhead (use helper funcs getCubeCoords(), xyz2idx())
        
        struct = 'flat'
        if struct == 'flat':
            self.V = np.zeros(self.xl*self.yl*self.zl, dtype=np.uint8)
        else:
            self.V = np.zeros((self.zl, self.yl, self.xl), dtype=np.uint8)
        
        # Create voxel array for grey values indicating hox many times a voxel
        # has been written
        self.contV = np.zeros(self.V.shape, dtype=np.uint8)
        
        # Create voxel array for bool values indicating if the voxel contains
        # raw data
        #self.usedV = np.zeros(self.V.shape, dtype=np.bool)  # it occupies as much as a np.uint8
        self.usedV = self.contV
        
        # Create voxel array for bool values indicating if the voxel belongs
        # to the sequence of slices
        self.internalV = np.zeros(self.V.shape, dtype=np.bool)
        
        
    
    def setUSImagesAlignmentParameters(self, **kwargs):
        """Set parameters for US scans alignement in global reference frame.
        See chapter :ref:`when-mem-error` for tips about setting these parameters.
        
        Parameters
        ----------
        wrapper : str
            Type of wrapper to create scanning silhouette.
            If 'parallelepipedon', the smallest wrapping paralellepipedon (with
            dimensions aligned with the global reference frame) is created between
            two US scans.
            If 'convex_hull', the convex hull is created between two US scans.
            This one is more accurate than 'parallelepipedon', but it takes more
            time to be created.
            If 'none' (default), no wrapper is created.
            
            .. image:: parall_vs_convexhull.png
               :scale: 50 %
        
        step : int
            Interval (in number of US frames) between two US scans
            used to create the wrapper. Default to 1.
        
        alignFrames : list
            List of frames (US time line) on which to perform US images alignment.
        
        fillVoxMethod : str
            Method for filling each voxel.
            If 'avg', an average between the current voxel value and the new value 
            is performed.
            If 'last', the new voxel value will replace the current one.
            If 'max', the highest voxel value will replace the current one.
        
        """
        
        # Check wrapper
        if 'wrapper' in kwargs:
            wrapper = kwargs['wrapper']
            checkWrapper(wrapper)
            self.wrapper = wrapper

        # Check step
        if 'step' in kwargs:
            step = kwargs['step']
            checkStep(step)
            self.step = step
        
        # Check frameWin
        if 'alignFrames' in kwargs:
            alignFrames = kwargs['alignFrames']
            checkIm2GlPose(self.R)
            checkAlignFrames(alignFrames, self.R.shape[0])
            self.alignFrames = alignFrames
            
        # Check fillVoxMethod
        if 'fillVoxMethod' in kwargs:
            fillVoxMethod = kwargs['fillVoxMethod']
            checkFillVoxMethod(fillVoxMethod)
            self.fillVoxMethod = fillVoxMethod
        
    
    def alignUSImages(self):
        """Align US images in the global reference frame.
        This task can take some time, and computation time is proportional
        to the *total* number of US images to align.
        
        """
        
        # Check input validity
        checkImDim(self.w)
        checkImDim(self.h)
        checkPixel2mm(self.pixel2mmX)
        checkPixel2mm(self.pixel2mmY)
        checkUsFiles(self.usFiles)
        checkIm2GlPose(self.R)
        checkGl2ConvPose(self.convR)
        checkFxyz([self.fx, self.fy, self.fz])
        # xl, xo
        checkV(self.V)
        checkV(self.contV)
        checkV(self.usedV)
        checkV(self.internalV)
        checkWrapper(self.wrapper)
        checkStep(self.step)
        checkFillVoxMethod(self.fillVoxMethod)

        # Create if necessary and check alignFrames
        if not self.alignFrames:
            self.alignFrames = range(0, self.R.shape[0])
        checkAlignFrames(self.alignFrames, self.R.shape[0])
        
        # Create pixel coordinates (in mm) in image reference frame
        print 'Creating pixel 3D coordinates in image reference frame ...'
        p = createImageCoords(self.h, self.w, self.pixel2mmY, self.pixel2mmX)
        print 'Pixel 3D coordinates calculated'
        
        # Calculate image corners coordinates
        pc = createImageCorners(self.w, self.h, self.pixel2mmX, self.pixel2mmY)
        
        # Calculate position for all the pixels, for all the time instant
        t = time.time()
        fileNames = self.usFiles
        ioffset = 0
        ivx = np.array(self.voxFrames)
        for f in xrange(0,len(fileNames)):
            # Read DICOM file
            print 'Reading DICOM file {0} ...'.format(fileNames[f])
            D, ds = readDICOM(fileNames[f])
            print 'DICOM file read'
            #print D.shape
            Ni = D.shape[1]
            iStart = None
            for i in xrange(0,Ni):
                # Calculate absolute index
                iR = i + ioffset
                # Check if frame has to be realigned
                if iR not in ivx:
                    continue
                if iR not in self.alignFrames:
                    continue
#                if iR not in self.validKineFrames:
#                    continue
                if iStart == None:
                    iStart = i
                # Create gray values
                I = pixelData2grey(D[:,i,:,:])
                print 'Inserting oriented slice for instant {0}/{1} ...'.format(iR+1,Ni)
                # Calculate frames position in space
                pg = np.dot(np.dot(self.convR,self.R[iR,:,:]),p) # mm
                x = (np.round(pg[0,:] * self.fx) - self.xo).squeeze() # 1 x Np
                y = (np.round(pg[1,:] * self.fy) - self.yo).squeeze()
                z = (np.round(pg[2,:] * self.fz) - self.zo).squeeze()
                # Fill voxel array with grey values
                idxV = xyz2idx(x, y, z, self.xl, self.yl, self.zl)
                if self.fillVoxMethod == 'avg':
                    #self.V[idxV] = (self.contV[idxV] * self.V[idxV]) / (self.contV[idxV] + 1) + I.ravel() / (self.contV[idxV] + 1)   # wrong: overflow 
                    self.V[idxV] = self.V[idxV] * (self.contV[idxV] / (self.contV[idxV] + 1)) + I.ravel() * (1. / (self.contV[idxV] + 1))  
                elif self.fillVoxMethod == 'last':
                    self.V[idxV] = I.ravel()
                elif self.fillVoxMethod == 'max':
                    self.V[idxV] = np.maximum(self.V[idxV], I.ravel()) 
                self.contV[idxV] += 1
                #self.usedV[idxV] = True                
                del I
                if self.wrapper == 'parallelepipedon':
                    xc = x
                    yc = y
                    zc = z
                elif self.wrapper == 'convex_hull':
                    # Calculate coordinates of image corners 
                    pcg = np.dot(np.dot(self.convR,self.R[iR,:,:]),pc) # mm
                    xc = (np.round(pcg[0,:] * self.fx) - self.xo).squeeze() # 1 x 4
                    yc = (np.round(pcg[1,:] * self.fy) - self.yo).squeeze()
                    zc = (np.round(pcg[2,:] * self.fz) - self.zo).squeeze()  
                else:
                    #self.internalV[idxV] = True
                    self.internalV[idxV] = True
                    continue
                # Create wrapper
                if i == iStart:
                    xcPrev = xc.copy()
                    ycPrev = yc.copy()
                    zcPrev = zc.copy()
                    continue
                if i < Ni-1:
                    if i % self.step:
                        continue
                if self.wrapper == 'parallelepipedon':
                    print 'Creating parallelepipedon ...'
                    xcMin, xcMax = np.min((xc.min(),xcPrev.min())), np.max((xc.max(),xcPrev.max()))
                    ycMin, ycMax = np.min((yc.min(),ycPrev.min())), np.max((yc.max(),ycPrev.max()))
                    zcMin, zcMax = np.min((zc.min(),zcPrev.min())), np.max((zc.max(),zcPrev.max()))
                    xcInternal, ycInternal, zcInternal = getCubeCoords(([xcMin,xcMax],[ycMin,ycMax],[zcMin,zcMax]))
                elif self.wrapper == 'convex_hull':
                    print 'Creating convex hull ...'
                    cCurrent = np.array((xc,yc,zc)).T
                    cPrev = np.array((xcPrev,ycPrev,zcPrev)).T
                    if not np.array_equal(cCurrent,cPrev):
                        cHull = np.vstack((cCurrent,cPrev))
                        cInternal = getCoordsInConvexHull(cHull)
                        xcInternal, ycInternal, zcInternal = cInternal[:,0], cInternal[:,1], cInternal[:,2]
                    else:
                        print 'The 2 slices are exactly overlapped. Impossible to create convex hull'
                #idxInternal = xyz2idx(xcInternal, ycInternal, zcInternal, self.xl, self.yl, self.zl).squeeze()
                idxInternal = xyz2idx(xcInternal, ycInternal, zcInternal, self.xl, self.yl, self.zl)
                self.internalV[idxInternal] = True
                xcPrev = xc.copy()
                ycPrev = yc.copy()
                zcPrev = zc.copy()
                
            ioffset += Ni
            del D, ds       
            
            
        #del self.contV
        self.usedV = self.contV.astype(np.bool)
        if self.wrapper <> 'none':
            del xcPrev, ycPrev, zcPrev, idxInternal, xcInternal, ycInternal, zcInternal
        elapsed = time.time() - t
        print 'Elapsed time: {0} s'.format(elapsed)
        idxEmptyN = np.sum(~self.usedV)
        pctEmpty = 100.0 * idxEmptyN / self.V.size
        print 'Pct of empty voxels: ({0}% total)'.format(pctEmpty)
        pctInternal = 100.0 * np.sum(self.internalV) / self.V.size
        print 'Estimate of pct of internal voxels: ({0}% total)'.format(pctInternal)
        if np.sum(self.internalV) > 0:
            pctInternalEmpty = 100.0 * np.sum(self.internalV & ~self.usedV) / np.sum(self.internalV)
        else:
            pctInternalEmpty = 0.
        print 'Estimate of pct of internal empty voxels: ({0}% internal)'.format(pctInternalEmpty)
    



    def setGapFillingParameters(self, **kwargs):
        """Set parameters for gap filling.
        
        Parameters
        ----------
        method : str
            Method for filling gaps.
            If 'VNN' (Voxel Nearest Neighbour, default), the nearest voxel to the gap is
            used to fill the gap. Arguments ``maxS`` and ``minPct`` will be ignored.
            If ``distTh` is set, voxels with a distance greater than this threshold will
            be ignored when filling gaps.
            If 'AVG_CUBE', this procedure is applied:
                1. create a cube with side 3 voxels, centered around the gap
                2. search for a minimum ``minPct`` percentage of non-gaps inside the cube (100% = number of voxels in the cube)
                3. if that percentage is found, a non-gap voxels average (wighted by the Euclidean distances) is performed into the cube
                4. if that percentage is not found, the cube size in incremented by 2 voxels
                5. if cube size is lesser than maxS, start again from point 2. Otherwise, stop and don't fill the gap.
            This method is much slower than 'VNN', but allows to limit the search area.
        
        maxS : int
            See ``method``. This number must be an odd number. Default to 1.
        
        minPct : float 
            See ``method``. This value must be between 0 and 1. Default to 0.
        
        blocksN : int
            Positive number (greater or equal than 1) indicating the number of
            subvoxel-arrays into which to decompose the gap-filling problem. This can be tuned to
            modify computation time and memory usage. Default to 100.
        
        blockDir : str
            String defining the direction for blocks motion.
            It can be 'X', 'Y', 'Z'.
        
        distTh : int
            See ``method``. This must be greater or equal than 1.
        
        Notes
        -----
        *Only* the gaps internal to the wrapper created by ``alighImages()`` will beconsidered.
        If a gap is not filled, its value will be considered the same as a *completely black* voxel.
        See chapter :ref:`when-mem-error` for tips about setting these parameters.
        
        """
        
        # Check method
        if 'method' in kwargs:
            method = kwargs['method']
            checkMethod(method)
            self.method = method
            
        # Check blocksN
        if 'blocksN' in kwargs:
            blocksN = kwargs['blocksN']
            checkBlocksN(blocksN)
            self.blocksN = blocksN

        # Check blockDir
        if 'blockDir' in kwargs:
            blockDir = kwargs['blockDir']
            checkBlockDir(blockDir)
            self.blockDir = blockDir

        # Check maxS
        if 'maxS' in kwargs:
            maxS = kwargs['maxS']
            checkMaxS(maxS)
            self.maxS = maxS
            
        # Check distTh
        if 'distTh' in kwargs:
            distTh = kwargs['distTh']
            checkDistTh(distTh)
            self.distTh = distTh
    
        
        # Check minPct
        if 'minPct' in kwargs:
            minPct = kwargs['minPct']
            checkMinPct(minPct)
            self.minPct = minPct
            
    
    def fillGaps(self):
        """Run the gap-filling procedure.
        This task can take some time.
        
        """
    
        # Check input validity
        checkMethod(self.method)
        checkBlocksN(self.blocksN)
        checkMaxS(self.maxS)
        checkBlockDir(self.blockDir)
        if self.method == 'VNN':
            checkDistTh(self.distTh)
        if self.method == 'AVG_CUBE':
            checkMinPct(self.minPct)        
        checkV(self.V)
        checkV(self.usedV)
        checkV(self.internalV)
        
        print 'Filling empty voxels ({0}), when possible ...'.format(self.method)
        if self.blockDir == 'X':
            bxl = np.ceil(self.xl / self.blocksN)
            byl = self.yl
            bzl = self.zl
        elif self.blockDir == 'Y':
            bxl = self.xl
            byl = np.ceil(self.yl / self.blocksN)
            bzl = self.zl
        elif self.blockDir == 'Z':
            bxl = self.xl
            byl = self.yl
            bzl = np.ceil(self.zl / self.blocksN)
#        blockSize = bxl * byl * bzl
        if len(self.V.shape) > 1:
            sliceMethod = 'fast'
        else:
            sliceMethod = 'slow'
        for b in xrange(0, self.blocksN):
            print 'Block {0} ...'.format(b+1)
            # Initialize block indices
            cLims = [None] * 3
            if self.blockDir == 'X':
                cLims[0] = [b*bxl, np.min([(b+1)*bxl,self.xl])]
                cLims[1] = [0, self.yl]
                cLims[2] = [0, self.zl]
                if (b+1)*bxl > self.xl:
                    bxl = self.xl - b * bxl
            elif self.blockDir == 'Y':
                cLims[0] = [0, self.xl]
                cLims[1] = [b*byl, np.min([(b+1)*byl,self.yl])]
                cLims[2] = [0, self.zl]
                if (b+1)*byl > self.yl:
                    byl = self.yl - b * byl
            elif self.blockDir == 'Z':
                cLims[0] = [0, self.xl]
                cLims[1] = [0, self.yl]
                cLims[2] = [b*bzl, np.min([(b+1)*bzl,self.zl])]
                if (b+1)*bzl > self.zl:
                    bzl = self.zl - b * bzl
            if sliceMethod == 'slow':
                xc, yc, zc = getCubeCoords(cLims)
                ind = xyz2idx(xc, yc, zc, self.xl, self.yl, self.zl)
                idxBlock = np.zeros(self.V.shape, dtype=np.bool)
                idxBlock[ind] = True
            if self.method == 'VNN':
                # Apply VNN
#                bzl = np.sum(idxBlock) / (bxl * byl)
                if sliceMethod == 'slow':
                    reshV = np.reshape((~self.usedV & self.internalV)[idxBlock], (bzl,byl,bxl))
                    reshV2 = np.reshape(self.V[idxBlock], (bzl,byl,bxl))
                elif sliceMethod == 'fast':
                    reshV = (~self.usedV & self.internalV)[cLims[2][0]:cLims[2][1],cLims[1][0]:cLims[1][1],cLims[0][0]:cLims[0][1]]
                    reshV2 = self.V[cLims[2][0]:cLims[2][1],cLims[1][0]:cLims[1][1],cLims[0][0]:cLims[0][1]]
                np.set_printoptions(threshold=np.nan)
                if self.distTh == None:
                    idxV = nd.distance_transform_edt(reshV, return_distances=False, return_indices=True)
                else:
                    edt, idxV = nd.distance_transform_edt(reshV, return_distances=True, return_indices=True)
                    idxTh = np.nonzero(edt > self.distTh)
                    idxV[0][idxTh] = idxTh[0]
                    idxV[1][idxTh] = idxTh[1]
                    idxV[2][idxTh] = idxTh[2]
                    del edt, idxTh
                if sliceMethod == 'slow':
                    self.V[idxBlock] = reshV2[tuple(idxV)].ravel()
                    self.usedV[idxBlock] = True
                    del idxBlock
                elif sliceMethod == 'fast':
                    self.V[cLims[2][0]:cLims[2][1],cLims[1][0]:cLims[1][1],cLims[0][0]:cLims[0][1]] = reshV2[tuple(idxV)]
                    self.usedV[cLims[2][0]:cLims[2][1],cLims[1][0]:cLims[1][1],cLims[0][0]:cLims[0][1]] = True
                del reshV, reshV2, idxV
                # Print some info
                pctInternalEmpty = 100.0 * np.sum(self.internalV & ~self.usedV) / np.sum(self.internalV)
                print '\tEstimate of pct of internal empty voxels: ({0}% internal)'.format(pctInternalEmpty) 
            elif self.method == 'AVG_CUBE':
                for S in np.arange(3, self.maxS+1, 2):
                    if b == 0:
                        # Generate voxel coordinates for the search cube
                        xCube, yCube, zCube = getCubeCoords(S)
                        # Remove central voxel of the cube
                        idxCentral = np.nonzero((xCube == 0) & (yCube == 0) & (zCube == 0))[0]
                        xCube = np.delete(xCube, idxCentral)[:, None]
                        yCube = np.delete(yCube, idxCentral)[:, None]
                        zCube = np.delete(zCube, idxCentral)[:, None]
                        # Calculate distance from each vixel to central voxel
                        distNeighs = (xCube**2 + yCube**2 + zCube**2)**(0.5)
                        idxSort = np.argsort(distNeighs)
                        distNeighs = 1. / distNeighs[idxSort,:]
                    idxEmpty = np.nonzero((~self.usedV) & idxBlock & self.internalV)[0]   # time bottleneck
                    # Get coordinates of empty voxels
                    xn, yn, zn = idx2xyz(idxEmpty, self.xl, self.yl, self.zl)
                    xn = np.tile(xn, (S**3-1,1))
                    yn = np.tile(yn, (S**3-1,1))
                    zn = np.tile(zn, (S**3-1,1))
                    idxNeighs = xyz2idx(xn+xCube,yn+yCube,zn+zCube, self.xl, self.yl, self.zl)
                    # Get values for neigbour voxels, empty or not
                    neighsV = self.V[idxNeighs]
                    neighsUsedV = self.usedV[idxNeighs]
                    del idxNeighs
                    # Sort by distance
                    neighsV = neighsV[idxSort,:]
                    neighsUsedV = neighsUsedV[idxSort,:]
                    # Fill some empty voxels
                    idxFillable = (np.sum(neighsUsedV, axis=0) >= np.round(self.minPct * (S**3-1)) ).squeeze()
                    wMeanNum = np.sum(neighsUsedV * neighsV * distNeighs, axis=0).squeeze()
                    wMeanDen = np.sum(neighsUsedV * distNeighs, axis=0).squeeze()
                    self.V[idxEmpty[idxFillable]] = (wMeanNum[idxFillable] / wMeanDen[idxFillable]).round().astype(np.uint8)  
                    self.usedV[idxEmpty[idxFillable]] = True   
                    # Print some info
                    pctInternalEmpty = 100.0 * np.sum(self.internalV & ~self.usedV) / np.sum(self.internalV)
                    print '\tEstimate of pct of internal empty voxels after filling with cube of side {0}: ({1}% internal)'.format(S, pctInternalEmpty)              
                    # Delete biggest arrays in inner loop                
                    del idxEmpty, neighsV, neighsUsedV, idxFillable, wMeanNum, wMeanDen
        
        print 'Empty voxels filled when possible'
        return pctInternalEmpty
        
    
    def getVoxelPhysicalSize(self):
        """Get physical size for a single voxel.
        
        Returns
        -------
        list
            3-elem list with voxel dimensions (in *mm*) for each direction.

        """
        
        # Check fxyz
        checkFxyz([self.fx, self.fy, self.fz])
        
        # Calculate physical dimensions (in mm)
        vx = 1. / self.fx
        vy = 1. / self.fy
        vz = 1. / self.fz
        
        return vx, vy, vz
    
    
    def setVtkImageDataProperties(self, **kwargs):
        """Set parameters of ``vtkImageData`` object.
        
        Whenever a ``vtkImageData`` has to be created (e.g. for exportation purpose)
        from the internal voxel-array structure, these parameters are used.
        
        Parameters
        ----------
        sxyz : mixed
            Spacing factors fot object.
            If list, it must contain 3 elements containing spacing factors for each voxel dimension (see `here <http://www.vtk.org/doc/nightly/html/classvtkImageData.html#ab3288d13810266e0b30ba0632f7b5b0b>`_).
            If 'auto', spacing factors are automatically calculated using scale factors ``fxyz`` (see method ``initVoxelArray()``)
            Each factor *s* is calculated by using the correspoding scale factor *f* as: s = LCM(fx,fy,fz) / f,
            where LCM is the Least Minimum Multiple operator.
        
        """

        # Check sxyz
        if 'sxyz' in kwargs:
            sxyz = kwargs['sxyz']
            checkSxyz(sxyz)
            if sxyz == 'auto':
                checkFxyz([self.fx, self.fy, self.fz])
                spacing = lcmm(self.fx, self.fy, self.fz) / np.array((self.fx, self.fy,self.fz))
                self.sx, self.sy, self.sz = spacing[0], spacing[1], spacing[2]
            else:
                self.sx, self.sy, self.sz = sxyz[0], sxyz[1], sxyz[2]
            print 'vtkImageData spacing factors set to: %d, %d, %d' % (self.sx, self.sy, self.sz)
        
    
    def exportVoxelArrayToVTI(self, outFile):
        """Export grey-values voxel-array to VTI file.
        
        VTI is a VTK file format (see `here <http://www.cacr.caltech.edu/~slombey/asci/vtk/vtk_formats.simple.html>`_).
        
        Parameters
        ----------
        outFile : str
            Full file path for the VTI file to be saved.
        
        """
        
        # Check input validity
        checkFilePath(outFile)
        checkSxyz([self.sx, self.sy, self.sz])
        checkV(self.V)
        # xl
        
        # Create vtkImageData object for grey values voxel array
        print 'Creating vtkImageData object for grey values voxel array...'
        vtkV = nparray2vtkImageData(self.V, (self.xl,self.yl,self.zl), (self.sx,self.sy,self.sz), vtk.VTK_UNSIGNED_CHAR)
        print 'vtkImageData object created'
        # Write grey values voxel array to file
        print 'Saving VTI file for grey values voxel array {0} ...'.format(outFile)
        vtkImageData2vti(outFile, vtkV)    
        print 'VTI file saved'

        
    
    def exportVoxelArraySilhouetteToVTI(self, outFile):
        """Export US scan silhouette voxel-array to VTI file.
        
        Parameters
        ----------
        outFile : str
            Full file path for the VTI file to be saved.
        
        """

        # Check input validity
        checkFilePath(outFile)
        checkSxyz([self.sx, self.sy, self.sz])
        checkV(self.internalV)
        
        # Create vtkImageData object for silhouette voxel array
        print 'Creating vtkImageData object for silhouette values voxel array...'
        vtkInternalV = nparray2vtkImageData(255*self.internalV.astype(np.uint8), (self.xl,self.yl,self.zl), (self.sx,self.sy,self.sz), vtk.VTK_UNSIGNED_CHAR)
        print 'vtkImageData object created'
        # Write silhouette voxel array to file
        print 'Saving VTI file for silhouette voxel array {0} ...'.format(outFile)
        vtkImageData2vti(outFile, vtkInternalV)    
        print 'VTI file saved'
        
        

    def calculateProbeCalibrationPrecision(self, prec='RP'):
        """Estimate calibration precision.
        
        Parameters
        ----------
        prec : str
            Precision type to estimate.
            If 'RP', Reconstruction Precision is estimated (see [Ref1]_). It needs
            single-point feature to be extracted for some US images of a calibration 
            quality assessment acquisition. The points are the reconstructed in 
            3D space, creating a cloud of points. RP is the mean of the distances
            between each 3D point and the 3D average point.
            
        """
        
        # Check input validity
        checkPrecType(prec)
        checkPixel2mm(self.pixel2mmX)
        checkPixel2mm(self.pixel2mmY)
        checkIm2GlPose(self.R)
        checkFeatures(self.features)
        
        # Calculate precision
        if prec == 'RP':
            print 'Calculating reconstruction precision...'
            precValue = calculateRP(self.R, self.pixel2mmX, self.pixel2mmY, self.features)            
            print 'Precision calculated'
        
        # Set data internally
        self.prec[prec] = precValue
        
    
    
    def getProbeCalibrationPrecision(self, prec='RP'):
        """Get estimated calibration precision data.
        
        Parameters
        ----------
        prec : str
            See method ``calculateProbeCalibrationPrecision()``.
            
        Returns
        -------
        float
            Precision estimation.
            
        """
        
        # Check input validity
        checkPrecType(prec)
        
        # Get precision
        if prec not in self.prec:
            raise Exception('This precision type was not calculated yet')
        
        return self.prec[prec]
        
        
    
    def calculateProbeCalibrationAccuracy(self, acc='DA', L=100., P=np.zeros((0,3))):
        """Estimate calibration accuracy.
        
        Parameters
        ----------
        acc : str
            Accuracy type to estimate.
            If 'DA', Distance Accuracy is estimated (see [Ref2]_). It needs
            2 single-point features to be extracted for some US images of a calibration 
            quality assessment acquisition. These 2 points (each for different US images)
            are reconstructed in global reference frame and the distance is calculated. This process can be
            repeated for other couples of US images. For instance, if one point is indicated
            for frames 1, 4, 10, 15, 25, 40, then 3 distances are calculated (1-4, 10-15, 25-40).
            DA is the mean of the difference between these distances and the gold-standard
            measured real distance ``L``.
            If 'RA', Reconstruction Accuracy is estimated (see [Ref2]_). It needs
            1 single-point feature to be extracted for some US images of a calibration 
            quality assessment acquisition. These points (each for different US images)
            are reconstructed in global reference frame. 
            RA is the mean of the norm of the difference between these points and 
            the gold-standard points ``P``.
            
        L : float
            Gold-standard distance (in *mm*) for DA estimation.
            
        P : np.ndarray
            Gold-standard 3D position (in *mm*) for RA estimation.
            It must be a N x 3 array containing 3D positions for points, where the time
            line is the same as the US data. Only the points whose time frames correspond
            to the single-point features.
            
        """
        
        # Check input validity
        checkAccType(acc)
        if acc == 'DA':    
            checkDist(L)
        checkPixel2mm(self.pixel2mmX)
        checkPixel2mm(self.pixel2mmY)
        checkIm2GlPose(self.R)
        checkFeatures(self.features)
        
        # Calculate precision
        if acc == 'DA':
            print 'Calculating distance accuracy...'
            accList, accValue = calculateDA(self.R, self.pixel2mmX, self.pixel2mmY, self.features, L)
        elif acc == 'RA':
            print 'Calculating reconstruction accuracy...'
            accList, accValue = calculateRA(self.R, self.pixel2mmX, self.pixel2mmY, self.features, P)            
        print 'Accuracy calculated'
        # Set data internally
        self.acc[acc] = (accList, accValue)
        


    def getProbeCalibrationAccuracy(self, acc='DA'):
        """Get estimated calibration accuracy data.
        
        Parameters
        ----------
        acc : str
            See method ``calculateProbeCalibrationAccuracy()``.
        
        Returns
        -------
        listDA : np.ndarray
            Array containing as many values as the keys into ``points``. 
            If 2 points where indicated in the corresponding US image, 
            than the value corresponds to the difference between the 
            points distance and ``L``, ``np.nan`` otherwise.
        
        DA : float
            Mean of ``listDA`` ignoring nans.

        """


        # Check input validity
        checkAccType(acc)
        
        # Get accuracy
        if acc not in self.acc:
            raise Exception('This accuracy type was not calculated yet')
        
        return self.acc[acc]   


    def calculateDevicesTimeDelay(self, method='vert_motion_sync', **kwargs): 
        """Estimate the delay between the US device and the optoelectronic device.
        
        Parameters
        ----------
        method: str
            Method used for the estimation.
            If 'vert_motion_sync', the user should have performed a vertical motion of the US probe
            so that the vertical coordinate of the markers cluster reference frame resambles a 
            sine wave. It is suggested to scan the bottom of a water tank and make sure that the
            bottom of the line is kept more or less horizontal. The center of that line should have
            been detected in advance. A cross-correlation, between the normalized y coordinate (in US
            image reference frame) of the line center and the the normalized vertical coordinate (in 
            global reference frame) of the origin of markers cluster reference frame, will be performed.
            Normalization consists of demeaning and dividing by the maximum of the rectified signal.
            From the cross-correlation signal, the maximum value within the time range (-1,+1), in 
            seconds, is found. The time instant in which that maximum occurs is the time delay estimation.
            If positive, the US device is early with respect to the optolectronic device. 
        
        vertCoordIdx : int, optional
            3D marker coordinate index representing the vertical coordinate with respect to global 
            reference frame (0 <= vertCoordIdx<= 2). Considered if ``method='vert_motion_sync'``.
        
        showGraphs : bool, optional
            If True, normalized signals to be correled and correlation signal will be displayed. 
            Execution will stop until the graphs windows is closed.
            Considered if ``method='vert_motion_sync'``.
        
        """
        
        # Check input validity
        checkTemporalCalibMethod(method)        
        if method == 'vert_motion_sync':
            checkPr2GlPose(self.Rpr, self.Tpr)
            checkFeatures(self.features)
        
        # Perform estimate
        if method == 'vert_motion_sync':
            # Get time vector
            timeVector = self.usTimeVector[0]
            # Get further arguments
            vertCoordIdx = kwargs['vertCoordIdx']
            showGraphs = kwargs['showGraphs']
            # Get vertical coordinate from opto device
            optoSignal = self.Tpr[:,vertCoordIdx]
            # Get height for point in image coordinates
            usSignal = np.zeros((optoSignal.shape[0],))
            for key, value in self.features.iteritems():
                usSignal[key] = value[0][1]
            # Estimate delay
            print 'Estimating time delay...'
            timeDelay = calculateTimeDelayXCorr(optoSignal, usSignal, 'Normalized height of markers cluster origin', 'Y coordinate for point features detected in US images', timeVector, 0.001, lagsBound=1., withPlots=showGraphs)
            print 'Time delay estimated'
            
        self.timeDelay = timeDelay
        
    
    def getDevicesTimeDelay(self):
        """Get estimated delay between the US device and the optoelectronic device
        (See method ``calculateDevicesTimeDelay()``).
        
        Returns
        -------
        float
            Time delay (in *seconds*).

        """

        
        return self.timeDelay
        
    
    def setDevicesTimeDelay(self, timeDelay):
        """Set delay between the US device and the optoelectronic device.

        Parameters
        ----------  
        timeDelay : float
            Time delay (in *seconds*) between the two devices. If positive, US device is early. 
        
        """
    
        checkTimeDelay(timeDelay)
        self.timeDelay = timeDelay
        
        
    def adjustUSTimeVector(self):
        """Adjust the original time vector of US images.
        The time delay set by ``setDevicesTimeDelay()`` will be subtracted from the
        original time vector extracted from US data.
        
        .. note::
        
            This method must be called before any method using optoelectronic data,
            such as ``calculatePoseForUSProbe()``.
        
        """
        
        # Check input validity
        checkTimeVector(self.usTimeVector)
        checkTimeDelay(self.timeDelay)
        
        # Adjust original time vectors
        print 'Adjusting US TimeVector for delay...'
        self.usTimeVectorAdj = copy.deepcopy(self.usTimeVector)
        for i in xrange(0, len(self.usTimeVectorAdj)):
            self.usTimeVectorAdj[i] = (np.array(self.usTimeVectorAdj[i]) - self.timeDelay).tolist()
        print 'TimeVector adjusted'
        
    
    def getAdjustedUSTimeVector(self):
        """Get adjusted US time vector (see ``adjustUSTimeVector()``).
        
        Returns
        -------
        list
            Adjusted US time vector.

        """
        
        return self.usTimeVectorAdj

        
        
        
        

        
        
        
        

        
        
        