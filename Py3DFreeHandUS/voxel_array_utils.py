# -*- coding: utf-8 -*-
"""
.. module:: voxel_array_utils
   :synopsis: helper module for voxel-array

"""

import numpy as np
from scipy.interpolate import griddata
import vtk
from vtk.util import numpy_support as nps


def getCoordsInConvexHull(p):
    """Create the convex hull for a list of points and the list of coorindates internal to it.
    
    Parameters
    ----------
    p : np.ndarray
        N x 3 list of coordinates for which to calculate the cinvex hull. Coordinates should be integer.
        
    Returns
    -------
    np.ndarray
        M x 3 array of coordinates, where M is the number of points internal to the convex hull.
    
    """
    # Get minimum and maximum coordinates
    xMin, yMin, zMin = p.min(axis=0)
    xMax, yMax, zMax = p.max(axis=0)
    # Create coordinates between maximim and minimum
    xCube, yCube, zCube = getCubeCoords(([xMin,xMax],[yMin,yMax],[zMin,zMax]))
    ci = np.array((xCube, yCube, zCube)).T
    # Linear interpolation 
    vi = griddata(p, np.ones((p.shape[0],)), ci, method='linear', fill_value=0)
    # Delete points outside the convex hull
    idx = np.nonzero(vi == 0)[0]
    cInternal = np.delete(ci, idx, axis=0)
    return cInternal

def getCubeCoords(S):
    """Create cube or parallelepipedon coordinates.
    
    Parameters
    ----------
    S : mixed
        Parallelepipedon or cube size.
        If int, it represents the cube side, and must be an odd number. 
        The coordinates origin is in the center of the cube.
        If list, it must contain 3 lists (for x, y and z), each one containing mininum and maximum coordinate values.
    
    Returns
    -------
    list
        List of 3 ``np.ndarray`` objects (for x, y and z), containing coordinate values into the parallelepipedon / cube.
    
    """
    if hasattr(S,'__len__') == False:
        l1, l2 = -(S-1)/2, (S+1)/2
        xx, yy, zz = np.mgrid[l1:l2,l1:l2,l1:l2]
    else:
        xx, yy, zz = np.mgrid[S[0][0]:S[0][1],S[1][0]:S[1][1],S[2][0]:S[2][1]]
    cx = xx.flatten()
    cy = yy.flatten()
    cz = zz.flatten()
    return cx, cy, cz
    
def idx2xyz(idx, xl, yl, zl):
    """Transform a list of indices of 1D array into coordinates of a 3D volume of certain sizes.
    
    Parameters
    ----------
    idx : np.ndarray
        1D array to be converted. An increment of ``idx``
        corresponds to a an increment of x. When reaching ``xl``, x is reset and 
        y is incremented of one. When reaching ``yl``, x and y are reset and z is
        incremented.
    
    xl, yl, zl : int 
        Sizes for 3D volume.
    
    Returns
    -------
    list 
        List of 3 ``np.ndarray`` objects (for x, y and z), containing coordinate value.

    """
    
    z = np.floor(idx / (xl*yl))
    r = np.remainder(idx, xl*yl)
    y = np.floor(r / xl)
    x = np.remainder(r, xl)
    return x, y, z

def xyz2idx(x, y, z, xl, yl, zl, idx='counter'):
    """Transform coordinates of a 3D volume of certain sizes into a list of indices of 1D array.
    This is the opposite of function ``idx2xyz()``.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinates to be converted.
        
    xl, yl, zl : int
        Sizes for 3D volume.
        
    idx: str
        Str ing indicating output type.
        If 'counter', the output is an array of voxel IDs, incrementing while x coordinate is incrementing.
        If 'list', a list (z,y,x) is created.
    
    Returns
    -------
    np.ndarray or list
        Voxel indices.

    """
    
    x[x >= xl] = xl-1
    y[y >= yl] = yl-1
    z[z >= zl] = zl-1
    x[x < 0] = 0
    y[y < 0] = 0
    z[z < 0] = 0
    if idx == 'counter':
        idx = (x + y * xl + z * (xl * yl)).astype(np.int32)
    elif idx == 'list':
        idx = (z.astype(np.int32), y.astype(np.int32), x.astype(np.int32))
    return idx

def nparray2vtkImageData(v, d, s, vtkScalarType):
    """Transform a 1D ``numpy`` array into ``vtk.vtkImageData`` object. 
    The object contains only one scalar component.
    
    Parameters
    ----------
    v : np.ndarray
        1D array to convert.
        
    d : list
        3-elem list of sizes of the ``vtk.vtkImageData``.
        
    s : list
        3-elem list of spacing factors of the ``vtk.vtkImageData`` (see `here <http://www.vtk.org/doc/nightly/html/classvtkImageData.html#ab3288d13810266e0b30ba0632f7b5b0b>`_).
    
    vtkScalarType : 
        Scalar type to be allocated (e.g. ``vtk.VTK_UNSIGNED_CHAR``).
    
    Returns
    -------
    vtk.vtkImageData
        object.

    """    
    
    # Create source
    source = vtk.vtkImageData()
    source.SetDimensions(d[0], d[1], d[2])
    source.SetNumberOfScalarComponents(1)
    source.SetScalarType(vtkScalarType)
    source.AllocateScalars()
    source.SetSpacing(s[0], s[1], s[2])
    # Copy numpy voxel array to vtkDataArray
    dataArray = nps.numpy_to_vtk(v, deep=0, array_type=None)
    source.GetPointData().GetScalars().DeepCopy(dataArray)
    return source

def vtkImageData2vti(filePath, source):
    """Export a ``vtk.vtkImageData`` object to VTI file.
    
    Parameters
    ----------
    filePath : str
        Full path for the VTI to be created.
        
    source : vtk.vtkImageData
        object.
    
    """
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filePath)
    writer.SetInput(source)
    writer.Write()