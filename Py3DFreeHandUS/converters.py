# -*- coding: utf-8 -*-
"""
.. module:: converters
   :synopsis: helper module for conversion between data formats.

"""

def vti2mat(fileIn, fileOut):
    """Convert voxel-array from VTI to MAT (MATLAB(R)) format.
    
    Parameters
    ----------
    fileIn : str
        Path for input VTI file.
        
    fileOut : str
        Path for output MAT file.
        
    """
    
    import numpy as np
    import vtk
    import scipy.io as sio
    from vtk.util import numpy_support as nps
    from math_utils import lcmm
    
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fileIn)
    reader.Update()
    vtkImageData = reader.GetOutput()
    dim = vtkImageData.GetDimensions()
    flatV = nps.vtk_to_numpy(vtkImageData.GetPointData().GetScalars())
    V = flatV.reshape(dim[::-1])
    spacing = np.array(vtkImageData.GetSpacing())[::-1]
    estimatedFactors = lcmm(*spacing) / spacing
    estimatedVoxelSize = 1. / estimatedFactors
    sio.savemat(fileOut, {'volume':V, 'spacing': spacing, 'estimated_voxel_size': estimatedVoxelSize})
    
    
def avi2dcm(fileIn, fileOut):
    """Convert gray-scale AVI file to gray-scale image-sequence DICOM file.
    Frame rate is *not* added as *CineRate* tag in the DICOM file.
    
    Parameters
    ----------
    fileIn : str
        Path for input AVI file.
        
    fileOut : str
        Path for output DICOM file.
        
    """
    
    import numpy as np
    import cv2
    import SimpleITK as sitk
    
    cap = cv2.VideoCapture(fileIn)
    Nf = int(cap.get(7))
    w = cap.get(3)
    h = cap.get(4)
    fps = cap.get(5)
    I = np.zeros((Nf,h,w), dtype=np.uint8)
    
    #while(cap.isOpened()):
    for i in xrange(Nf):
        print 'Converting frame %d ...' % i
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        I[i,:,:] = gray
    
    cap.release()
    image = sitk.GetImageFromArray(I)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(fileOut)
    writer.Execute(image)
    
    
def arr2aviOCV(fileName, M, fps):
    """Convert gray-scale Numpy 3D image array to AVI file (use OpenCV).
    
    Parameters
    ----------
    fileName : str
        Path for output AVI file.
        
    M : np.ndarray(uint8)
        F x H x W 3D array, representing a sequence of F images, each H x W.
        
    fps : int
        frame rate for the output file.
        
    """

    import numpy as np
    import cv2    
    
    # Define the codec and create VideoWriter object
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    fourcc = cv2.cv.CV_FOURCC(*'msvc') # it seems the only one working 
    #fourcc = -1 # choose codec manually 
    try:
        out = cv2.VideoWriter(fileName,fourcc, fps, M.shape[::-1][:-1])
        for frame in M:
            #frame = cv2.flip(frame,0)
            # write the flipped frame
            out.write(frame)
    except:
        out.release()
        
    # Release everything if job is finished
    out.release()
    
    
def arr2aviMPY(fileName, M, fps):
    """Convert gray-scale Numpy 3D image array to AVI file (use moviepy).
    
    Parameters
    ----------
    fileName : str
        Path for output AVI file.
        
    M : np.ndarray(uint8)
        F x H x W 3D array, representing a sequence of F images, each H x W.
        
    fps : int
        frame rate for the output file.
        
    """
    import numpy as np
    from moviepy.editor import ImageSequenceClip
    
    D = [np.dstack([m] * 3) for m in M]
    clip = ImageSequenceClip(D, fps=fps)    
    clip.write_videofile(fileName, codec='mpeg4', ffmpeg_params=['-vb','1M'])
    