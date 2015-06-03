# -*- coding: utf-8 -*-
"""
.. module:: image_utils
   :synopsis: helper module for image data handling

"""

import numpy as np
import dicom
import cv2
import SimpleITK as sitk


def createImageCorners(w, h, pixel2mmX, pixel2mmY):
    """Create corner coordinates for an image.
    Top-left corner is supposed to be the (0, 0) corner.
    
    Parameters
    ----------
    w : int
        Image width (in *pixel*)
        
    h : int
        Image height (in *pixel*)
        
    pixel2mmX, pixel2mmY : float
        Number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*)

    Returns
    -------
    np.ndarray
        4 x 4 array of coordinates. Each column is a corner. To (x, y), (z, 1)
        are also added to make them ready to be mulitplied by a roto-translation
        matrix.
    
    """
    pc = np.array((
            (w*pixel2mmX,0,0,1),
            (w*pixel2mmX,h*pixel2mmY,0,1),
            (0,0,0,1),
            (0,h*pixel2mmY,0,1),
        )).T
    return pc
    

def createImageCoords(h, w, pixel2mmY, pixel2mmX): 
    """Create all pixel coordinates for an image.
    Top-left corner is supposed to be the (0, 0) corner.
    
    Parameters
    ----------
    w : int
        Image width (in *pixel*)
        
    h : int
        Image height (in *pixel*)
        
    pixel2mmX, pixel2mmY : float
        Number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*)
        
    Returns
    -------
    np.ndarray
        4 x (w * h) array of coordinates. Each column is a point. To (x, y), (z, 1)
        are also added to make them ready to be mulitplied by a roto-translation
        matrix.
    
    """     
    Np = h * w
    x = np.linspace(0,w-1,w) * pixel2mmX
    y = np.linspace(0,h-1,h) * pixel2mmY
    xv, yv = np.meshgrid(x, y)
    xv = np.reshape(xv.ravel(), (1,Np))
    yv = np.reshape(yv.ravel(), (1,Np))
    zv = np.zeros((1,Np))
    b = np.ones((1,Np))
    p = np.concatenate((xv,yv,zv,b), axis=0) # 4 x Np
    return p
    

def createCenteredMaskCoords(cx, cy, h, w):
    """Create all pixel coordinates for a centered mask around a point.
    Center point is (cx, cy).
    
    Parameters
    ----------
    cx : int
        X coordinate for mask center.
        
    cy : int
        Y coordinate for mask center.  
    
    w : int
        Mask width. Should be odd.
        
    h : int
        Mask height. Should be odd.
        
    Returns
    -------
    np.ndarray
        (w * h) x 2 array of coordinates. Each row is a point.
    
    """   
    x = (np.linspace(-(w-1)/2,(w-1)/2,w) + cx).astype(np.int32)
    y = (np.linspace(-(h-1)/2,(h-1)/2,h) + cy).astype(np.int32)
    xv, yv = np.meshgrid(x, y)
    C = np.array([xv.ravel(), yv.ravel()]).T.astype(np.float32)
    return C


def pixelData2grey(D):
    """Convert pixel array to grey values.
    
    Parameters
    ----------
    D : np.ndarray 
        Pixel array, in format Nch x Nf x Nr x Nc, to convert.
        If Nch is 3, then channels are supposed to be R, G, B.
        If Nch is 2, then the values are supposed to be grey level and alpha.
        If Nch is 1, then the values are supposed to be grey level.
    
    Returns
    -------
    np.ndarray
        Nf x Nr x Nc array of grey level.

    """
    d = D.shape
    if d[0] < 3:
        I = D[0,:]
    elif d[0] == 3:
        I = rgb2grey(D[0,:], D[1,:], D[2,:])
    else:
        raise Exception('Image data format not recognized')
    return I
        
    
    
def rgb2grey(R, G, B):
    """Convert RGB channels to grey levels, by using the formula `here <http://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems>`_.
    
    Parameters
    ----------
    R, G, B : np.ndarray
        Arrays containing red, green and blue values. R,G, B must have the same dimensions.
    
    Returns
    -------
    np.ndarray
        Array of grey levels, having the same dimensions of either R, G or B.
    
    """
    I = (.2126*R+.7152*G+.0722*B).astype(np.uint8)
    return I


def readDICOM(filePath, method='flattened'):
    """Read DICOM file (containing data for Nc channels, Nf frames, and images of size Nr x Nc).
    
    Parameters
    ----------
    filePath : str
        DICOM full file path.
        
    method : str
        Pixel array parsing method.
        If 'RGB', pixel array is supposed to be 3 x Nf x Nr x Nc. Data for frame i is into [:,i,:,:].
        If 'flattened', pixel array is supposed to be Nch x Nf x Nr x Nc. Data for frame i is into [j,k:k+Nch,:,:], where j = floor(Nch*i / Nf), k = (Nch*i) % Nf.
        When using 'flattened', pixel array with dimension Nf x Nr x Nc is also supprted (the only stored value is supposed to be a grey level).
    
    Returns
    -------
    D : np.ndarray
        Pixel array reshaped in the standard way Nch x Nf x Nr x Nc as for ``method='RGB'``.
    
    ds : dicom.dataset.FileDataset
        Additional parameters in the DICOM file.
    
    """
    ds = dicom.read_file(filePath)
    D = ds.pixel_array
    if method == 'RGB':
        pass
    if method == 'flattened':
        if len(D.shape) == 4: # first dimension keeps or data channels (RGB, grey-alpha, grey, ...)
            N = D.shape[0]
            D = np.reshape(D, (D.shape[0]*D.shape[1],D.shape[2],D.shape[3]))[None,:]
            D = D[:,::N,:,:]
        elif len(D.shape) == 3:
            D = D[None,:]
        else:
            raise Exception('{0}: unknown data format'.format(method))
    return D, ds
    

def readSITK(filePath):
    """Helper for reading SITK-compatible input file
    
    Parameters
    ----------
    filePath : str
        Full file path.

    Returns
    -------
    I : np.ndarray
        Pixel array (dimensions depend on input dimensions).
    
    image : sitk.Image
        Image object as read by SITK.
    
    """
    
    # Read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(filePath)
    image = reader.Execute()
    
    # Convert images to Numpy arrays 
    I = sitk.GetArrayFromImage(image)
    
    return I, image
    

def NCC(I1, I2):
    """Calculate Normalized Cross-Correlation between 2 binary images.
    
    Parameters
    ----------
    I1, I2 : np.ndarray(uint8)
        The 2 binary images, same size is required.
    
    Returns
    -------
    float
        NCC.
    
    """
    m1 = np.mean(I1)
    s1 = np.std(I1)
    m2 = np.mean(I2)
    s2 = np.std(I2)
    if s1 == 0 or s2 == 0:
        NCC = 0.
    else:
        demI1 = I1 - m1
        demI2 = I2 - m2
        prodI = demI1 * demI2
        prodI *= 1. / (s1 * s2)
        NCC = np.mean(prodI)
    return NCC
    
    

def createWhiteMask(frameGray, cx, cy, h, w):
    """Create white mask in a grayscale frame, centered around (cx, cy).
    
    Parameters
    ----------
    frameGray : np.ndarray
        frame to copy from for creating a new one with the mask    
    
    cx : int
        X coordinate for mask center.
        
    cy : int
        Y coordinate for mask center.  
    
    w : int
        Mask width. Should be odd.
        
    h : int
        Mask height. Should be odd.

    Returns
    -------
    tuple
        First element is the frame containing the white mask, and black around.
        Second element is the mask coordinates.
    
    """
    new_mask = np.empty_like(frameGray)
    new_mask[:] = 0
    pts = createCenteredMaskCoords(round(cx), round(cy), h, w)
    a, b = pts[:,0].min(), pts[:,1].min()
    c, d = pts[:,0].max(), pts[:,1].max()
    new_mask2 = cv2.rectangle(new_mask, (a,b),(c,d), 255, -1)
    if new_mask2 is not None:
        new_mask = new_mask2
    return new_mask, pts
    

def findCornersInMask(frameGray, cx, cy, h, w, featureParams):
    """Find Shi-Tomasi corners in a subpart of a frame.
    The research mask is centered around (cx, cy).
    
    Parameters
    ----------
    frameGray : np.ndarray
        frame to search corners from.    
    
    cx : int
        X coordinate for mask center.
        
    cy : int
        Y coordinate for mask center.  
    
    w : int
        Mask width. Should be odd.
        
    h : int
        Mask height. Should be odd.
    
    featureParams : dict
        See **kwargs in ``cv2.cv2.goodFeaturesToTrack()``.

    Returns
    -------
    tuple
        First element is a N x 1 x 2 array containing coordinates of N good 
        corners to track. Second element being the frame contanining the mask.
    
    """
    new_mask, pts = createWhiteMask(frameGray, cx, cy, h, w)
    new_p0 = cv2.goodFeaturesToTrack(frameGray, mask=new_mask, **featureParams)
    return new_p0, new_mask
    


    
    
    
    