# -*- coding: utf-8 -*-
"""
.. module:: calib
   :synopsis: Module for US probe calibration (equations creation, solution, solution quality assessment)

"""

import numpy as np    
from sympy import Matrix, Symbol, var
from sympy import cos as c, sin as s
from scipy.optimize import root, minimize
from sympy.utilities.lambdify import lambdify
from scipy import stats
from voxel_array_utils import *
from image_utils import *
from segment import singlePointFeaturesTo3DPointsMatrix
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
try:
    from calib_c import *
except:
    warnings.warn('Module calib_c was not found. Open a console, cd to the Py3DFreeHandUS folder, and type: python cython_setup.py build_ext --inplace --compiler=mingw32', stacklevel=2)


def MatrixOfMatrixSymbol(b, r, c):
    """Emulates ``Matrix(MatrixSymbol(b, r, c))`` as for SymPy version 0.7.2-.
    
    From SymPy 0.7.5+, ``M = Matrix(MatrixSymbol(b, r, c))`` is not safe since:
    
    - the format of symbols has changed fomr s_ij to s[i, j]
    - the method ``M.inv()`` is broken
    
    """
    return Matrix(r, c, lambda i,j: var('%s_%d%d' % (b,i,j)))   # inspired by: http://stackoverflow.com/questions/6877061/automatically-populating-matrix-elements-in-sympy
    

def creatCalibMatrix():
    """Generate and return symbolic expression of 4 x 4 affine rotation matrix from US probe reference frame to US image reference frame.

    Returns
    -------
    prTi : sympy.matrices.matrices.MutableMatrix 
        The matrix expression.
        
    syms : list
        List of ``sympy.core.symbol.Symbol`` symbol objects used to generate the expression. 
    
    """
    
    x1 = Symbol('x1')
    y1 = Symbol('y1')
    z1 = Symbol('z1')
    alpha1 = Symbol('alpha1')
    beta1 = Symbol('beta1')
    gamma1 = Symbol('gamma1')
    
    prTi = Matrix(([c(alpha1)*c(beta1), c(alpha1)*s(beta1)*s(gamma1)-s(alpha1)*c(gamma1), c(alpha1)*s(beta1)*c(gamma1)+s(alpha1)*s(gamma1), x1],\
                   [s(alpha1)*c(beta1), s(alpha1)*s(beta1)*s(gamma1)+c(alpha1)*c(gamma1), s(alpha1)*s(beta1)*c(gamma1)-c(alpha1)*s(gamma1), y1],\
                   [-s(beta1), c(beta1)*s(gamma1), c(beta1)*c(gamma1), z1],\
                   [0, 0, 0, 1]\
    ))
    
    syms = [x1, y1, z1, alpha1, beta1, gamma1]
    
    return prTi, syms


def createCalibEquations():
    """Generate and return symbolic calibration equations (1) in [Ref2]_.
    
    Returns
    -------
    Pph : sympy.matrices.matrices.MutableMatrix
        3 x 1 matrix containing symbolic equations (1) in [Ref2]_.
    
    J : sympy.matrices.matrices.MutableMatrix*)
        3 x 14 matrix representing the Jacobian of equations ``Pph``.
    
    prTi : sympy.matrices.matrices.MutableMatrix*)
        4 x 4 affine rotation matrix from US probe reference frame to US image reference frame.
    
    syms : dict
        Dictionary of where keys are variable names and values are ``sympy.core.symbol.Symbol`` objects. 
        These symbols were used to create equations in ``Pph``, ``J``, ``prTi``.
    
    variables : list
        14-elem list of variable names (see ``process.Process.calibrateProbe()``).
    
    mus : list
        14-elem list of varables measurement units.
    
    """
    
    # Pi
    sx = Symbol('sx')
    sy = Symbol('sy')
    u = Symbol('u')
    v = Symbol('v')
    Pi = Matrix(([sx * u],\
                 [sy * v],\
                 [0],\
                 [1]\
    ))
    
    # prTi
    prTi, syms = creatCalibMatrix()
    [x1, y1, z1, alpha1, beta1, gamma1] = syms
    
    # wTpr
    #wTpr = Matrix(MatrixSymbol('wTpr', 4, 4))
    wTpr = MatrixOfMatrixSymbol('wTpr', 4, 4)
    wTpr[3, 0:4] = np.array([0,0,0,1])
    
    # phTw
    x2 = Symbol('x2')
    y2 = Symbol('y2')
    z2 = Symbol('z2')
    alpha2 = Symbol('alpha2')
    beta2 = Symbol('beta2')
    gamma2 = Symbol('gamma2')
    
    phTw = Matrix(([c(alpha2)*c(beta2), c(alpha2)*s(beta2)*s(gamma2)-s(alpha2)*c(gamma2), c(alpha2)*s(beta2)*c(gamma2)+s(alpha2)*s(gamma2), x2],\
                   [s(alpha2)*c(beta2), s(alpha2)*s(beta2)*s(gamma2)+c(alpha2)*c(gamma2), s(alpha2)*s(beta2)*c(gamma2)-c(alpha2)*s(gamma2), y2],\
                   [-s(beta2), c(beta2)*s(gamma2), c(beta2)*c(gamma2), z2],\
                   [0, 0, 0, 1]\
    ))
    
    # Calculate full equations
    Pph = phTw * wTpr * prTi * Pi
    Pph = Pph[0:3,:]
    
    # Calculate full Jacobians 
    x = Matrix([sx, sy, x1, y1, z1, alpha1, beta1, gamma1, x2, y2, z2, alpha2, beta2, gamma2])
    J = Pph.jacobian(x)
    
    # Create symbols dictionary
    syms = {}
    for expr in Pph:
        atoms = list(expr.atoms(Symbol))
        for i in xrange(0, len(atoms)):
            syms[atoms[i].name] = atoms[i]
    
    # Create list of variables and measurements units
    variables = ['sx', 'sy', 'x1', 'y1', 'z1', 'alpha1', 'beta1', 'gamma1', 'x2', 'y2', 'z2', 'alpha2', 'beta2', 'gamma2']
    mus = ['mm/px', 'mm/px', 'mm', 'mm', 'mm', 'rad', 'rad', 'rad', 'mm', 'mm', 'mm', 'rad', 'rad', 'rad']
        
    # Return data
    return Pph, J, prTi, syms, variables, mus
    

def solveCalibEquations(eq, J, syms, variables, init, xtol, ftol, Rpr, Tpr, features, regJ):
    """Solve calibration equations (1) in [Ref2]_. More specifically, a system
    of non-linear equations is created by coyping the symbolic equation ``eq``,
    replacing the experimental data for each time frame, and stacking it in the
    system to be solved. The iterative method used to solve the system is *Levenberg–Marquardt*.
    
    Parameters
    ----------
    eq : sympy.core.add.Add
         Template equation to be stacked in the system.
    
    J : sympy.core.add.Add
        Jacobian of equation ``eq``.
    
    syms : dict
        Dictionary of where keys are variable names and values are ``sympy.core.symbol.Symbol`` objects. These symbols were used to create equations in ``eq``, ``J``, ``prTi``.
    
    variables : list
        List of variable names (see ``process.Process.calibrateProbe()``).
    
    init : list
        List of initial values (same order of ``variables``).
    
    xtol : float 
        Relative error desired in the approximate solution (see argument ``options['xtol']`` or ``tol`` in ``scipy.optimize.root()``).
        
    ftol : float 
        Relative error desired in the sum of squares (see argument ``options['ftol']`` in ``scipy.optimize.root()``).    
    
    Rpr : np.ndarrayN x 3 x 3 array, where ``R[i,:,:]`` represents the rotation matrix from the US probe reference frame to the global reference frame, for time frame ``i``.
    
    Tpr : np.ndarray
        N x 3 array, where ``Tpr[i,:]`` represents the vector from the global reference frame origin to the US probe reference frame origin, for time frame ``i``.
    
    features: dict
        Dictionary where keys are frame numbers and values are lists of tuples, each one representing a point position in the corresponding US image.
    
    regJ: bool
        If True, Jacobian will be regularized by using scaling in eq 15, [Ref2]_. 
    
    Returns
    -------
    sol : scipy.optimize.Result
        Solution object (see ``scipy.optimize.root``).
    
    k : int
        Condition number (see [Ref2]_).

    """
     
    # Check variables number vs equations number
    if len(features) * len(features[features.keys()[0]]) < len(variables):
        raise Exception('The number of equations must be breater than the number of variables')
        
    # Lambdify expressions
    xLam = []
    for k in xrange(0,len(variables)):
        xLam.append(syms[variables[k]])
    xLam.append(syms['wTpr_00'])
    xLam.append(syms['wTpr_01'])
    xLam.append(syms['wTpr_02'])
    xLam.append(syms['wTpr_03'])
    xLam.append(syms['wTpr_10'])
    xLam.append(syms['wTpr_11'])
    xLam.append(syms['wTpr_12'])
    xLam.append(syms['wTpr_13'])
    xLam.append(syms['wTpr_20'])
    xLam.append(syms['wTpr_21'])
    xLam.append(syms['wTpr_22'])
    xLam.append(syms['wTpr_23'])
    xLam.append(syms['u'])
    xLam.append(syms['v'])
    eqLam = lambdify(xLam, eq)
    JLam = lambdify(xLam, J)
    
    # Create solver function
    global jac
    jac = None
    scales = np.ones((len(variables)))
    
    def funcSolver(x, syms, variables, eq, J, Rpr, Tpr, points):
        frames = points.keys()
        f = []
        df = []
        for i in xrange(0,len(frames)):
            # Initialize arguments for lambdified functions
            args = []   # for Jacobian
            args2 = []  # for equation
            for k in xrange(0,len(variables)):
                args.append(x[k])
                args2.append(x[k] / scales[k])
            # Get time frame
            fr = int(frames[i])
            # Substitute wTpr (constant)
            args.append(Rpr[fr,0,0])
            args.append(Rpr[fr,0,1])
            args.append(Rpr[fr,0,2])
            args.append(Tpr[fr,0])
            args.append(Rpr[fr,1,0])
            args.append(Rpr[fr,1,1])
            args.append(Rpr[fr,1,2])
            args.append(Tpr[fr,1])
            args.append(Rpr[fr,2,0])
            args.append(Rpr[fr,2,1])
            args.append(Rpr[fr,2,2])
            args.append(Tpr[fr,2])
            args2.append(Rpr[fr,0,0])
            args2.append(Rpr[fr,0,1])
            args2.append(Rpr[fr,0,2])
            args2.append(Tpr[fr,0])
            args2.append(Rpr[fr,1,0])
            args2.append(Rpr[fr,1,1])
            args2.append(Rpr[fr,1,2])
            args2.append(Tpr[fr,1])
            args2.append(Rpr[fr,2,0])
            args2.append(Rpr[fr,2,1])
            args2.append(Rpr[fr,2,2])
            args2.append(Tpr[fr,2])
            for j in xrange(0,len(points[frames[i]])):
                # Substitute u, v (constant)
                args.append(points[frames[i]][j][0])
                args.append(points[frames[i]][j][1])
                args2.append(points[frames[i]][j][0])
                args2.append(points[frames[i]][j][1])
                # Evaluate equation
                fNew = eqLam(*args2)
                dfNew = np.array(JLam(*args))[0].tolist()
                del args[-2:]
                del args2[-2:]
                f.append(fNew)
                df.append(dfNew)
                
        # Save Jacobian for later use
        global jac
        jac = np.array(df)
        # Return equtions and jacobian
        print 'Solving {0} equations ...'.format(len(f))
        return f, df

        
    # Solve equations system
    if regJ:
        # Create regularization matrix
        f0, df0 = funcSolver(init, syms, variables, eq, J, Rpr, Tpr, features)
        J0 = np.array(df0)
        for i in xrange(0, J0.shape[1]):
            # Get scale factors
            scales[i] = np.linalg.norm(J0[:,i])
            # Regularize Jacobian
            J[:,i] /= scales[i]
            # Re-initialize variables for regularization
            init[i] *= scales[i]
        # Re-lambdify Jacobian
        JLam = lambdify(xLam, J)        
        
    options = {}
    
    if xtol is not None:
        options['xtol'] = xtol
    if ftol is not None:
        options['ftol'] = ftol

    sol = root(funcSolver, init, jac=True, method='lm', args=(syms,variables,eq,J,Rpr,Tpr,features), options=options)

    if regJ:
        # Unscale solution
        for i in xrange(0, J0.shape[1]):
            sol.x[i] /= scales[i]
    
    # Calculate problem condition number
    U, s, V = np.linalg.svd(jac)
    kond = int(s[0] / s[-1])
    return sol, kond
    
    
def createCalibExpressionsForMaxNCC():
    """Generate and return symbolic calibration roto-translation matrix in (3) in [Ref3]_.
    
    Returns
    -------
    i2Ti1 : sympy.matrices.matrices.MutableMatrix*)
        4 x 4 matrix containing symbolic roto-translation matrix in (3) in [Ref3]_.
    
    prTi : sympy.matrices.matrices.MutableMatrix*)
        4 x 4 affine rotation matrix from US probe reference frame to US image reference frame.
    
    syms : dict
        Dictionary of where keys are variable names and valuea are ``sympy.core.symbol.Symbol`` objects. These symbols
        were used to create equations in ``i2Ti1``.
    
    variables : list
        6-elem list of variable names (see ``process.Process.calibrateProbe()``).
    
    mus : list
        6-elem list of varables measurement units.

    """
    
    
    # prTi
    prTi, syms = creatCalibMatrix()
    [x1, y1, z1, alpha1, beta1, gamma1] = syms
    
    # wTpr1
    #wTpr1 = Matrix(MatrixSymbol('wTpr1', 4, 4))
    wTpr1 = MatrixOfMatrixSymbol('wTpr1', 4, 4)
    wTpr1[3, 0:4] = np.array([0,0,0,1])
    
    # wTpr2
    #wTpr2 = Matrix(MatrixSymbol('wTpr2', 4, 4))
    wTpr2 = MatrixOfMatrixSymbol('wTpr2', 4, 4)
    wTpr2[3, 0:4] = np.array([0,0,0,1])
    
    # i2Ti1
    i2Ti1 = prTi.inv() * wTpr2.inv() * wTpr1 * prTi
    
    # Create symbols dictionary
    syms = {}
    for expr in i2Ti1:
        atoms = list(expr.atoms(Symbol))
        for i in xrange(0, len(atoms)):
            syms[atoms[i].name] = atoms[i]
    
    # Create list of variables and measurements units
    variables = ['x1', 'y1', 'z1', 'alpha1', 'beta1', 'gamma1']
    mus = ['mm', 'mm', 'mm', 'rad', 'rad', 'rad']
        
    # Return data
    return i2Ti1, prTi, syms, variables, mus   
    


def maximizeNCCint(i2Ti1, syms, variables, init, Rpr, Tpr, I, pixel2mmX, pixel2mmY, frames, thZ):
    """(*Deprecated*) Minimize a modification of expression (1) in [Ref3]_. More specifically, it aims at maximizing the average Normalized Cross-Correlation of the intersection of pair of US images.
    
    Parameters
    ----------
    i2Ti1 : sympy.core.add.Add
        4 x 4 matrix containing symbolic roto-translation matrix in (3) in [Ref3]_.
    
    syms : dict
        Dictionary of where keys are variable names and values are ``sympy.core.symbol.Symbol`` objects. These symbols were used to create equations in ``eq``, ``J``, ``prTi``.
    
    variables : list
        List of variable names (see ``process.Process.calibrateProbe()``).
    
    init : list
        List of initial values (same order of ``variables``).
    
    Rpr : np.ndarray
        N x 3 x 3 array, where ``R[i,:,:]`` represents the rotation matrix from the US probe reference frame to the global reference frame, for time frame ``i``.
    
    Tpr : np.ndarray
        N x 3 array, where ``Tpr[i,:]`` represents the vector from the global reference frame origin to the US probe reference frame origin, for time frame ``i``.
    
    I : np.ndarray 
        N x Nr x Nc array, representing image data.
    
    pixel2mmX, pixel2mmY : float
        Number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*).

    frames : list
        Each element must be a list of 2 elements, representing a frames combination for NCC calculation.
    
    thZ : float
        Threshold value (in *mm*) under which points can be considered belonging to an image plane.
    
    Returns
    -------
    scipy.optimize.Result
        Solution object (see ``scipy.optimize.minimize``).  

    """

    # Create all pixel coordinates (in mm) for image 1
    Nf, h, w = I.shape
    p1 = createImageCoords(h, w, pixel2mmY, pixel2mmX)
    
    # Covert coordinates from mm to pixels
    c1 = p1.copy()
    c1[0,:] = np.round(p1[0,:] / pixel2mmX)
    c1[1,:] = np.round(p1[1,:] / pixel2mmY)
    c1[2,:] = np.zeros((p1.shape[1],))
    
    # Lambdify expressions
    xLam = []
    for k in xrange(0,len(variables)):
        xLam.append(syms[variables[k]])
    xLam.append(syms['wTpr1_00'])
    xLam.append(syms['wTpr1_01'])
    xLam.append(syms['wTpr1_02'])
    xLam.append(syms['wTpr1_03'])
    xLam.append(syms['wTpr1_10'])
    xLam.append(syms['wTpr1_11'])
    xLam.append(syms['wTpr1_12'])
    xLam.append(syms['wTpr1_13'])
    xLam.append(syms['wTpr1_20'])
    xLam.append(syms['wTpr1_21'])
    xLam.append(syms['wTpr1_22'])
    xLam.append(syms['wTpr1_23'])
    xLam.append(syms['wTpr2_00'])
    xLam.append(syms['wTpr2_01'])
    xLam.append(syms['wTpr2_02'])
    xLam.append(syms['wTpr2_03'])
    xLam.append(syms['wTpr2_10'])
    xLam.append(syms['wTpr2_11'])
    xLam.append(syms['wTpr2_12'])
    xLam.append(syms['wTpr2_13'])
    xLam.append(syms['wTpr2_20'])
    xLam.append(syms['wTpr2_21'])
    xLam.append(syms['wTpr2_22'])
    xLam.append(syms['wTpr2_23'])
    i2Ti1Lam = lambdify(xLam, i2Ti1)
    
    # Minimize expression
    def funcSolver(x, syms, variables, i2Ti1, Rpr, Tpr, I):
        
        f = 0.
        for fr in frames:
            
            # Get frame pair
            fr1 = fr[0]
            fr2 = fr[1]
            
            # Substitue current solution estimate
            args = []
            for k in xrange(0,len(variables)):
                args.append(x[k])

            # Substitute wTpr1
            args.append(Rpr[fr1,0,0])
            args.append(Rpr[fr1,0,1])
            args.append(Rpr[fr1,0,2])
            args.append(Tpr[fr1,0])
            args.append(Rpr[fr1,1,0])
            args.append(Rpr[fr1,1,1])
            args.append(Rpr[fr1,1,2])
            args.append(Tpr[fr1,1])
            args.append(Rpr[fr1,2,0])
            args.append(Rpr[fr1,2,1])
            args.append(Rpr[fr1,2,2])
            args.append(Tpr[fr1,2])
            # Substitute wTpr2
            args.append(Rpr[fr2,0,0])
            args.append(Rpr[fr2,0,1])
            args.append(Rpr[fr2,0,2])
            args.append(Tpr[fr2,0])
            args.append(Rpr[fr2,1,0])
            args.append(Rpr[fr2,1,1])
            args.append(Rpr[fr2,1,2])
            args.append(Tpr[fr2,1])
            args.append(Rpr[fr2,2,0])
            args.append(Rpr[fr2,2,1])
            args.append(Rpr[fr2,2,2])
            args.append(Tpr[fr2,2])
    
            # Evaluate roto-translation matrix
            T = np.array(i2Ti1Lam(*args))
            
            # Calculate coordinates in image 2
            p2 = np.dot(T, p1)
            
            # Calculate the indices for the intersecting coordinates
            idxC = np.abs(p2[2,:]) < thZ  # mm
            
            # Check if there is no intersection
            if np.sum(idxC) == 0:
                continue
            
            # Convert coordinates from mm to pixels
            c2 = p2.copy()
            c2[0,:] = np.round(p2[0,:] / pixel2mmX)
            c2[1,:] = np.round(p2[1,:] / pixel2mmY)
            c2[2,:] = np.zeros((p2.shape[1],))
            
            # Create indices for assignment
            idxI1 = xyz2idx(c2[0,idxC], c2[1,idxC], c2[2,idxC], w, h, 1).astype(np.int32)
            idxIfr1 = xyz2idx(c1[0,idxC], c1[1,idxC], c1[2,idxC], w, h, 1).astype(np.int32)
            
            # Prepare images for cross-correlation
            I2 = I[fr2,:,:]
            I1 = np.zeros(h * w, dtype=I.dtype)
            I1[idxI1] = I[fr1,:,:].ravel()[idxIfr1]
            I1 = I1.reshape((h, w))
            
            # Calculate NCC between the 2 images  (I2 the original, I1 the reconstruction)
            ncc = NCC(I1, I2)
        
            # Update expression to minimize
            f -= ncc
        
        f /= len(frames)

        # Return value to minimize
        print 'Minimizing expression (current value: %f) ...' % f
        return f
        
    sol = minimize(funcSolver, init, method='Nelder-Mead', args=(syms,variables,i2Ti1,Rpr,Tpr,I))
    
    return sol
    

def maximizeNCC(i2Ti1, syms, variables, init, Rpr, Tpr, I, pixel2mmX, pixel2mmY, frames, savePath, thZ, maxExpr, mask=None):
    """Minimize a modification of expression (1) in [Ref3]_. More specifically, it aims at maximizing the average Normalized Cross-Correlation of the intersection of pair of US images.
    
    Parameters
    ----------
    i2Ti1 : sympy.core.add.Add: 
        4 x 4 matrix containing symbolic roto-translation matrix in (3) in [Ref3]_.
    
    syms : dict
        Dictionary of where keys are variable names and values are ``sympy.core.symbol.Symbol`` objects. These symbols were used to create equations in ``eq``, ``J``, ``prTi``.
    
    variables : list
        List of variable names (see ``process.Process.calibrateProbe()``).
    
    init : list
        List of initial values (same order of ``variables``).
    
    Rpr : np.ndarray
        N x 3 x 3 array, where ``R[i,:,:]`` represents the rotation matrix from the US probe reference frame to the global reference frame, for time frame ``i``.
    
    Tpr : np.ndarray
        N x 3 array, where ``Tpr[i,:]`` represents the vector from the global reference frame origin to the US probe reference frame origin, for time frame ``i``.
    
    I : np.ndarray
        N x Nr x Nc array, representing image data.
    
    pixel2mmX, pixel2mmY : float
        Number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*).
    
    frames : list
        A 2-elem list where the first element is a list of original images sweep frames and the second element is a 2-elem list defining start and end frame of the reconstruction sweep.      
    
    savePath : str
        if not empty, it will be used to save each the couple original image - reconstruction for each iteration. 
        Each file name is in the format it<itn>_im<ofn>.jpeg, where <itn> is the iteration number (for Nelder-Mead method), <ofn> is the original image frame number.  
    
    thZ : float
        Threshold value (in *mm*) under which points on a reconstruction sweep can be considered belonging to an original image plane.
    
    maxExpr : str
        Expression to maximize.
        If 'avg_NCC', the NCCs calculated for each wanted pair original frame vs reconstruction template will be averaged.
        If 'weighted_avg_NCC', the NCCs calculated for each wanted pair original frame vs reconstruction template will be averaged using as weigths the percentage of reconstructed template.
        This percentage, in the bottom-left picture in the figures saved in ``savePath``, corresponds to the ratio between the area occupied by the straight lines and the image size.
        
    mask : mixed 
        Mask defining a sub-part of the original images to be considered.
        If None, the whole part of the original images will be considered.
        Otherwise, it must be an No x Nr x Nc array, where No is the number of original images.
    
    Returns
    -------
    scipy.optimize.Result
        solution object (see ``scipy.optimize.minimize``).  

    """

    # Create all pixel coordinates (in mm) for image 1
    Nf, h, w = I.shape
    p1 = createImageCoords(h, w, pixel2mmY, pixel2mmX)
    
    # Covert coordinates from mm to pixels
    c1 = p1.astype(np.int32)
    c1[0,:] = np.round(p1[0,:] / pixel2mmX)
    c1[1,:] = np.round(p1[1,:] / pixel2mmY)
    c1[2,:] = 0.
    
    # Lambdify expressions
    xLam = []
    for k in xrange(0,len(variables)):
        xLam.append(syms[variables[k]])
    xLam.append(syms['wTpr1_00'])
    xLam.append(syms['wTpr1_01'])
    xLam.append(syms['wTpr1_02'])
    xLam.append(syms['wTpr1_03'])
    xLam.append(syms['wTpr1_10'])
    xLam.append(syms['wTpr1_11'])
    xLam.append(syms['wTpr1_12'])
    xLam.append(syms['wTpr1_13'])
    xLam.append(syms['wTpr1_20'])
    xLam.append(syms['wTpr1_21'])
    xLam.append(syms['wTpr1_22'])
    xLam.append(syms['wTpr1_23'])
    xLam.append(syms['wTpr2_00'])
    xLam.append(syms['wTpr2_01'])
    xLam.append(syms['wTpr2_02'])
    xLam.append(syms['wTpr2_03'])
    xLam.append(syms['wTpr2_10'])
    xLam.append(syms['wTpr2_11'])
    xLam.append(syms['wTpr2_12'])
    xLam.append(syms['wTpr2_13'])
    xLam.append(syms['wTpr2_20'])
    xLam.append(syms['wTpr2_21'])
    xLam.append(syms['wTpr2_22'])
    xLam.append(syms['wTpr2_23'])
    i2Ti1Lam = lambdify(xLam, i2Ti1)
    
    No = len(frames[0])
    
    # Create full mask, if not existing
    if mask == None:
        mask = np.ones((No, h, w), dtype=np.bool)
    
    global cont
    cont = 0
    
    # Minimize expression
    def funcSolver(x, syms, variables, i2Ti1, Rpr, Tpr, I):
        
        # Print data for current iteration
        _printIterVariables(variables, init, x)

        global cont
        cont += 1
        NCCs = np.zeros((No,))
        pctIs = np.zeros((No,))
        
        for fr2, i in zip(frames[0], range(0, No)):

            I1 = np.zeros((h,w), dtype=I.dtype)
            I1mask = np.zeros((h,w), dtype=np.bool)
            I2 = I[fr2,:,:]
            I2rays = I2.copy()
            
            for fr1 in xrange(frames[1][0], frames[1][1]):
                
                # Substitue current solution estimate
                args = []
                for k in xrange(0,len(variables)):
                    args.append(x[k])
    
                # Substitute wTpr1
                args.append(Rpr[fr1,0,0])
                args.append(Rpr[fr1,0,1])
                args.append(Rpr[fr1,0,2])
                args.append(Tpr[fr1,0])
                args.append(Rpr[fr1,1,0])
                args.append(Rpr[fr1,1,1])
                args.append(Rpr[fr1,1,2])
                args.append(Tpr[fr1,1])
                args.append(Rpr[fr1,2,0])
                args.append(Rpr[fr1,2,1])
                args.append(Rpr[fr1,2,2])
                args.append(Tpr[fr1,2])
                # Substitute wTpr2
                args.append(Rpr[fr2,0,0])
                args.append(Rpr[fr2,0,1])
                args.append(Rpr[fr2,0,2])
                args.append(Tpr[fr2,0])
                args.append(Rpr[fr2,1,0])
                args.append(Rpr[fr2,1,1])
                args.append(Rpr[fr2,1,2])
                args.append(Tpr[fr2,1])
                args.append(Rpr[fr2,2,0])
                args.append(Rpr[fr2,2,1])
                args.append(Rpr[fr2,2,2])
                args.append(Tpr[fr2,2])
        
                # Evaluate roto-translation matrix
                T = np.array(i2Ti1Lam(*args))
                
                # Calculate coordinates in image 2
                p2 = np.dot(T, p1)
                
                # Calculate the indices for the intersecting coordinates
                idxC = np.abs(p2[2,:]) < thZ  # mm
                
                # Check if there is intersection
                if np.sum(idxC) == 0:
                    continue
                
                # Convert coordinates from mm to pixels
                c2 = p2.astype(np.int32)
                c2[0,:] = np.round(p2[0,:] / pixel2mmX)
                c2[1,:] = np.round(p2[1,:] / pixel2mmY)
                
                # Check if the intersection is within image size
                if np.sum(idxC) == 0:
                    continue
                
                # Create indices for reconstrucete image
                r, c = c2[1,idxC], c2[0,idxC]
                r[r < 0] = 0
                r[r >= h] = h-1
                c[c < 0] = 0
                c[c >= w] = w-1
                
                # Update reconstruction image
                I1[r, c] = I[fr1, c1[1,idxC], c1[0,idxC]]
                
                # Update reconstruction image mask
                I1mask[r, c] = mask[i, r, c]
                
                # Add intersections on original image
                I2rays[r, c] = np.iinfo(I.dtype).max
                
                
            # Calculate percentage of image information from reconstruction
            #pctI = 100. * I1mask.sum() / np.prod(I1mask.shape)
            pctI = 100. * I1mask.sum() / mask[i,:,:].sum()
            print 'Percentage of reconstructed image #%d: %.2f' % (fr2, pctI)
                
            # Calculate NCC between the 2 images  (I2 the original, I1 the reconstruction)
            ncc = NCC(I1[I1mask], I2[I1mask])
            
            # Save each couple image - reconstruction to file
            if len(savePath) <> 0:
                plt.close('all')
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(I2 * mask[i,:,:])
                plt.title('Original image (#%d)' % fr2)
                plt.subplot(2,2,2)
                plt.imshow(I1 * mask[i,:,:])
                plt.title('Reconstruction')
                plt.subplot(2,2,3)
                plt.imshow(I2rays)
                plt.title('Original image + intersections')
                plt.suptitle('NCC = %.2f' % ncc)
                plt.tight_layout()
                plt.savefig(savePath + '/it%d_im%d.jpeg' % (cont,fr2))
        
            # Store NCCs and reconstruction percentage
            NCCs[i] = ncc
            pctIs[i] = pctI
        
        # Update expression to minimize
        NCCg = NCCs[~np.isnan(NCCs)]
        pctIg = pctIs[~np.isnan(NCCs)]
        if maxExpr == 'avg_NCC':
            f = -np.nanmean(NCCg)
        elif maxExpr == 'weighted_avg_NCC':
            f = -np.average(NCCg, weights=pctIg)

        # Return value to minimize
        print 'Minimizing expression (current value: %f) ...' % f
        return f
        
    sol = minimize(funcSolver, init, method='Nelder-Mead', args=(syms,variables,i2Ti1,Rpr,Tpr,I))
    
    return sol
    
    

def maximizeNCCcy(i2Ti1, syms, variables, init, Rpr, Tpr, I, pixel2mmX, pixel2mmY, frames, savePath, thZ, maxExpr, mask=None):
    """Same as ``maximizeNCC()``, but with Cython implementation.
    Needs compilation first!
    """

    # Create all pixel coordinates (in mm) for image 1
    Nf, h, w = I.shape
    p1 = createImageCoords(h, w, pixel2mmY, pixel2mmX)
    
    # Covert coordinates from mm to pixels
    c1 = p1.astype(np.int32)
    c1[0,:] = np.round(p1[0,:] / pixel2mmX)
    c1[1,:] = np.round(p1[1,:] / pixel2mmY)
    c1[2,:] = 0.
    
    # Lambdify expressions
    xLam = []
    for k in xrange(0,len(variables)):
        xLam.append(syms[variables[k]])
    xLam.append(syms['wTpr1_00'])
    xLam.append(syms['wTpr1_01'])
    xLam.append(syms['wTpr1_02'])
    xLam.append(syms['wTpr1_03'])
    xLam.append(syms['wTpr1_10'])
    xLam.append(syms['wTpr1_11'])
    xLam.append(syms['wTpr1_12'])
    xLam.append(syms['wTpr1_13'])
    xLam.append(syms['wTpr1_20'])
    xLam.append(syms['wTpr1_21'])
    xLam.append(syms['wTpr1_22'])
    xLam.append(syms['wTpr1_23'])
    xLam.append(syms['wTpr2_00'])
    xLam.append(syms['wTpr2_01'])
    xLam.append(syms['wTpr2_02'])
    xLam.append(syms['wTpr2_03'])
    xLam.append(syms['wTpr2_10'])
    xLam.append(syms['wTpr2_11'])
    xLam.append(syms['wTpr2_12'])
    xLam.append(syms['wTpr2_13'])
    xLam.append(syms['wTpr2_20'])
    xLam.append(syms['wTpr2_21'])
    xLam.append(syms['wTpr2_22'])
    xLam.append(syms['wTpr2_23'])
    dummy0 = Symbol('dummy0')
    dummy1 = Symbol('dummy1')
    for col in xrange(3):
        i2Ti1[3,col] = dummy0
    i2Ti1[3,3] = dummy1
    xLam.append(dummy0)
    xLam.append(dummy1)
    i2Ti1Lam = lambdify(xLam, i2Ti1, [{'ImmutableMatrix': np.array}, 'numpy'])
    
    # Calculate slices indices
    fr2 = np.array(frames[0], dtype=np.int)
    N2 = fr2.shape[0]
    fr1 = np.arange(frames[1][0], frames[1][1], dtype=np.int)
    N1 = fr1.shape[0]
    idxFr1 = np.tile(fr1, N2)
    idxFr2 = np.repeat(fr2, [N1]*N2)
    
    # Extract wTpr1
    RprFr1_00 = Rpr[idxFr1,0,0]
    RprFr1_01 = Rpr[idxFr1,0,1]
    RprFr1_02 = Rpr[idxFr1,0,2]
    TprFr1_0 = Tpr[idxFr1,0]
    RprFr1_10 = Rpr[idxFr1,1,0]
    RprFr1_11 = Rpr[idxFr1,1,1]
    RprFr1_12 = Rpr[idxFr1,1,2]
    TprFr1_1 = Tpr[idxFr1,1]
    RprFr1_20 = Rpr[idxFr1,2,0]
    RprFr1_21 = Rpr[idxFr1,2,1]
    RprFr1_22 = Rpr[idxFr1,2,2]
    TprFr1_2 = Tpr[idxFr1,2]
    
    # Extract wTpr2
    RprFr2_00 = Rpr[idxFr2,0,0]
    RprFr2_01 = Rpr[idxFr2,0,1]
    RprFr2_02 = Rpr[idxFr2,0,2]
    TprFr2_0 = Tpr[idxFr2,0]
    RprFr2_10 = Rpr[idxFr2,1,0]
    RprFr2_11 = Rpr[idxFr2,1,1]
    RprFr2_12 = Rpr[idxFr2,1,2]
    TprFr2_1 = Tpr[idxFr2,1]
    RprFr2_20 = Rpr[idxFr2,2,0]
    RprFr2_21 = Rpr[idxFr2,2,1]
    RprFr2_22 = Rpr[idxFr2,2,2]
    TprFr2_2 = Tpr[idxFr2,2]
    
    dummy0v = np.zeros((N1*N2,))
    dummy1v = np.ones((N1*N2,))
    
    # Create full mask, if not existing
    if mask == None:
        mask = np.ones((N2, h, w), dtype=np.bool)
        
    global cont
    cont = 0
    
    # Minimize expression
    def funcSolver(x, syms, variables, i2Ti1, Rpr, Tpr, I):
        
        # Print data for current iteration
        _printIterVariables(variables, init, x)
        global cont
        cont += 1        
        
        # Substitue current solution estimate
        args = []
        for k in xrange(0,len(variables)):
            args.append(np.tile(x[k], N1*N2))

        # Substitute wTpr1
        args.append(RprFr1_00)
        args.append(RprFr1_01)
        args.append(RprFr1_02)
        args.append(TprFr1_0)
        args.append(RprFr1_10)
        args.append(RprFr1_11)
        args.append(RprFr1_12)
        args.append(TprFr1_1)
        args.append(RprFr1_20)
        args.append(RprFr1_21)
        args.append(RprFr1_22)
        args.append(TprFr1_2)
        
        # Substitute wTpr2
        args.append(RprFr2_00)
        args.append(RprFr2_01)
        args.append(RprFr2_02)
        args.append(TprFr2_0)
        args.append(RprFr2_10)
        args.append(RprFr2_11)
        args.append(RprFr2_12)
        args.append(TprFr2_1)
        args.append(RprFr2_20)
        args.append(RprFr2_21)
        args.append(RprFr2_22)
        args.append(TprFr2_2)
        
        args.append(dummy0v)
        args.append(dummy1v)

        # Evaluate roto-translation matrix
        T = np.array(i2Ti1Lam(*args))   # 4 x 4 x N
        
        # Rotate matrix ot make it N x 4 x 4
        T = T.transpose((1,0,2)).transpose((2,1,0))

        # Calculate NCCs andtheir weights
        NCCs, pctIs = compoundNCC(I, mask.astype(np.uint8), fr1, fr2, T, p1, c1, thZ, pixel2mmX, pixel2mmY, cont, savePath)
        
        # Update expression to minimize
        NCCg = NCCs[~np.isnan(NCCs)]
        pctIg = pctIs[~np.isnan(NCCs)]
        if maxExpr == 'avg_NCC':
            f = -np.nanmean(NCCg)
        elif maxExpr == 'weighted_avg_NCC':
            f = -np.average(NCCg, weights=pctIg)

        # Return value to minimize
        print 'Minimizing expression (current value: %f) ...' % f
        return f
        
    sol = minimize(funcSolver, init, method='Nelder-Mead', args=(syms,variables,i2Ti1,Rpr,Tpr,I))
    
    return sol
    
    

def maximizeNCCfast(i2Ti1, syms, variables, init, Rpr, Tpr, I, pixel2mmX, pixel2mmY, frames, savePath, thZ, maxExpr):
    """(*Deprecated*) Same as ``maximizeNCC()``, but with vectorized implementation.
    Despite it is about 2x faster than ``maximizeNCC()``, we experienced that the this function is intensive memory-wise, so we suggest not to use it yet.
    """

    # Create all pixel coordinates (in mm) for image 1
    Nf, h, w = I.shape
    p1 = createImageCoords(h, w, pixel2mmY, pixel2mmX)
    
    # Covert coordinates from mm to pixels
    c1 = p1.copy()
    c1[0,:] = np.round(p1[0,:] / pixel2mmX)
    c1[1,:] = np.round(p1[1,:] / pixel2mmY)
    c1[2,:] = 0
    
    # Lambdify expressions
    xLam = []
    for k in xrange(0,len(variables)):
        xLam.append(syms[variables[k]])
    xLam.append(syms['wTpr1_00'])
    xLam.append(syms['wTpr1_01'])
    xLam.append(syms['wTpr1_02'])
    xLam.append(syms['wTpr1_03'])
    xLam.append(syms['wTpr1_10'])
    xLam.append(syms['wTpr1_11'])
    xLam.append(syms['wTpr1_12'])
    xLam.append(syms['wTpr1_13'])
    xLam.append(syms['wTpr1_20'])
    xLam.append(syms['wTpr1_21'])
    xLam.append(syms['wTpr1_22'])
    xLam.append(syms['wTpr1_23'])
    xLam.append(syms['wTpr2_00'])
    xLam.append(syms['wTpr2_01'])
    xLam.append(syms['wTpr2_02'])
    xLam.append(syms['wTpr2_03'])
    xLam.append(syms['wTpr2_10'])
    xLam.append(syms['wTpr2_11'])
    xLam.append(syms['wTpr2_12'])
    xLam.append(syms['wTpr2_13'])
    xLam.append(syms['wTpr2_20'])
    xLam.append(syms['wTpr2_21'])
    xLam.append(syms['wTpr2_22'])
    xLam.append(syms['wTpr2_23'])
    i2Ti1Lam = lambdify(xLam, i2Ti1)
    
    global cont
    cont = 0
    
    # Minimize expression
    def funcSolver(x, syms, variables, i2Ti1, Rpr, Tpr, I):
        
        # Print data for current iteration
        _printIterVariables(variables, init, x)

        global cont
        cont += 1
        No = len(frames[0])
        NCCs = np.zeros((No,))
        pctIs = np.zeros((No,))
        
        for fr2, i in zip(frames[0], range(0, No)):

            I1 = np.zeros((h,w), dtype=I.dtype)
            I1mask = np.zeros((h,w), dtype=np.bool)
            I2 = I[fr2,:,:]
            I2rays = I2.copy()
            T = np.zeros((frames[1][1]-frames[1][0], 4, 4))
            #Ir = I[frames[1][0]:frames[1][1],:,:]
            
            for fr1 in xrange(frames[1][0], frames[1][1]):
                
                # Substitue current solution estimate
                args = []
                for k in xrange(0,len(variables)):
                    args.append(x[k])
    
                # Substitute wTpr1
                args.append(Rpr[fr1,0,0])
                args.append(Rpr[fr1,0,1])
                args.append(Rpr[fr1,0,2])
                args.append(Tpr[fr1,0])
                args.append(Rpr[fr1,1,0])
                args.append(Rpr[fr1,1,1])
                args.append(Rpr[fr1,1,2])
                args.append(Tpr[fr1,1])
                args.append(Rpr[fr1,2,0])
                args.append(Rpr[fr1,2,1])
                args.append(Rpr[fr1,2,2])
                args.append(Tpr[fr1,2])
                # Substitute wTpr2
                args.append(Rpr[fr2,0,0])
                args.append(Rpr[fr2,0,1])
                args.append(Rpr[fr2,0,2])
                args.append(Tpr[fr2,0])
                args.append(Rpr[fr2,1,0])
                args.append(Rpr[fr2,1,1])
                args.append(Rpr[fr2,1,2])
                args.append(Tpr[fr2,1])
                args.append(Rpr[fr2,2,0])
                args.append(Rpr[fr2,2,1])
                args.append(Rpr[fr2,2,2])
                args.append(Tpr[fr2,2])
        
                # Evaluate roto-translation matrix
                T[fr1-frames[1][0],:,:] = np.array(i2Ti1Lam(*args))
                
                
            # Calculate coordinates in image 2
            p2 = np.dot(T, p1)
            del T
            
            # Calculate the indices for the intersecting coordinates
            idxInt = np.argwhere(np.abs(p2[:,2,:]) < thZ)  # mm
            
            # Convert coordinates from mm to pixels
            c2 = p2
            c2[:,0,:] = np.round(p2[:,0,:] / pixel2mmX)
            c2[:,1,:] = np.round(p2[:,1,:] / pixel2mmY)
            
            # Create indices for fast assignment
            Fr = idxInt[:,0]
            C = idxInt[:,1]
            R1 = c1[1,C].astype(np.int32)
            C1 = c1[0,C].astype(np.int32)
            Rr = c2[Fr,1,C].ravel().astype(np.int32)
            Rr[Rr < 0.] = 0
            Rr[Rr >= h] = h-1
            Cr = c2[Fr,0,C].ravel().astype(np.int32)
            Cr[Cr < 0.] = 0
            Cr[Cr >= w] = w-1
            idxI1 = [Rr,Cr]
            idxIr = [Fr+frames[1][0],R1,C1]
            
            # Update reconstruction image
            I1[idxI1] = I[idxIr]
            
            # Update reconstruction image mask
            I1mask[idxI1] = True

            # Add intersections on original image
            I2rays[idxI1] = np.iinfo(I.dtype).max
            del R1, C1, Fr, Rr, Cr, p2
                
            # Calculate percentage of image information from reconstruction
            pctI = 100. * I1mask.sum() / np.prod(I1mask.shape)
            print 'Percentage of reconstructed image #%d: %.2f' % (fr2, pctI)
                
            # Calculate NCC between the 2 images  (I2 the original, I1 the reconstruction)
            ncc = NCC(I1[I1mask], I2[I1mask])
            
            # Save each couple image - reconstruction to file
            if len(savePath) <> 0:
                plt.close('all')
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(I2)
                plt.title('Original image (#%d)' % fr2)
                plt.subplot(2,2,2)
                plt.imshow(I1)
                plt.title('Reconstruction')
                plt.subplot(2,2,3)
                plt.imshow(I2rays)
                plt.title('Original image + intersections')
                plt.suptitle('NCC = %.2f' % ncc)
                plt.tight_layout()
                plt.savefig(savePath + '/it%d_im%d.jpeg' % (cont,fr2))
        
            # Store NCCs and reconstruction percentage
            NCCs[i] = ncc
            pctIs[i] = pctI
        
        # Update expression to minimize
        NCCg = NCCs[~np.isnan(NCCs)]
        pctIg = pctIs[~np.isnan(NCCs)]
        if maxExpr == 'avg_NCC':
            f = -np.nanmean(NCCg)
        elif maxExpr == 'weighted_avg_NCC':
            f = -np.average(NCCg, weights=pctIg)

        # Return value to minimize
        print 'Minimizing expression (current value: %f) ...' % f
        return f
        
    sol = minimize(funcSolver, init, method='Nelder-Mead', args=(syms,variables,i2Ti1,Rpr,Tpr,I))
    
    return sol

    

def calculateRP(T, sx, sy, points):
    """Calculate point reconstruction reconstruction precision, as in [Ref1]_. 
    It needs single-point feature to be extracted for some US images of a calibration 
    quality assessment acquisition. The points are the reconstructed in 
    3D space, creating a cloud of points. RP is the mean of the distances
    between each 3D point and the 3D average point.
    
    Parameters
    ----------
    T : np.ndarray
        N x 4 x 4 array where ``T[i,:,:]`` represents the roto-translation matrix from US image reference frame to global reference frame, for time frame ``i``.
    
    sx, sy : float
        Number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*).
    
    points : dict
        dictionary where keys are frame numbers and values are lists of tuples, each one representing a point position in the corresponding US image.
    
    Returns
    -------
    float
        Reconstruction precision value

    """
    
    # Calculate points in the global reference frame
    pointsGl = np.zeros((4,len(points)))
    frames = np.sort(points.keys())
    for i in xrange(0, len(points)):
        if len(points[frames[i]]) <> 1:
            pointsGl[:,i] = np.nan
            continue
        fr = frames[i]
#        point = np.zeros((4,))
#        point[0] = points[frames[i]][0][0] * sx
#        point[1] = points[frames[i]][0][1] * sy
#        point[2:4] = (0, 1)
        point = singlePointFeaturesTo3DPointsMatrix(points, sx, sy, idx=(fr,)).squeeze()
        pointsGl[:,i] = np.dot(T[fr,:,:],point)
    pointsGl = pointsGl[0:3,:]
    
    # Calculate RP
    meanPoint = stats.nanmean(pointsGl, axis=1)
    pointsNorm = np.zeros((pointsGl.shape[1],))
    for i in xrange(0,len(pointsNorm)): # np.linalg.norm has no 'axis' argument for NumPy 1.7.1
        pointsNorm[i] = np.linalg.norm(pointsGl[:,i]-meanPoint)
    RP = np.mean(pointsNorm)
    return RP


def calculateDA(T, sx, sy, points, L):
    """Calculate Distance Accuracy, as indicated in [Ref2]_. It needs
    2 single-point features to be extracted for some US images of a calibration 
    quality assessment acquisition. These 2 points (each for different US images)
    are reconstructed in and the distance is calculated. This process can be
    repeated for other couples of US images. For instance, if one point is indicated
    for frames 1, 4, 10, 15, 25, 40, then 3 distances are calculated (1-4, 10-15, 25-40).
    DA is the mean of the difference between these distances and the gold-standard
    measured real distance ``L``.
    
    Parameters
    ----------
    T : np.ndarray
        N x 4 x 4 array where ``T[i,:,:]`` represents the roto-translation matrix from US image reference frame to global reference frame, for time frame ``i``.
    
    sx, sy : float
        Number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*).
    
    points : dict
        Dictionary where keys are frame numbers and values are lists of tuples, each one representing a point position in the corresponding US image. Only one tuple is needed.
    
    L : float
        Gold-standard distance (in *mm*) for distance accuracy estimation.   
    
    Returns
    -------
    listDA : np.ndarray
        Array containing distances, each on calculated by using points from 2 consecutive frame numbers from ``points``.
    
    DA : float
        Mean of ``listDA`` ignoring nans.
    
    """

    # Calculate points in the global reference frame
    Nc = np.floor(len(points) / 2.)
    pointsGl1 = np.zeros((4,Nc)) * np.nan
    pointsGl2 = np.zeros((4,Nc)) * np.nan
    frames = np.sort(points.keys())
    c = 0
    for i in xrange(0, len(points)-1, 2):
        if len(points[frames[i]]) <> 1 or len(points[frames[i+1]]) <> 1:
            continue
        fr1 = frames[i]
        fr2 = frames[i+1]
#        point1 = np.zeros((4,))
#        point1[0] = points[frames[i]][0][0] * sx
#        point1[1] = points[frames[i]][0][1] * sy
#        point1[2:4] = (0, 1)
#        pointsGl1[:,c] = np.dot(T[fr1,:,:],point1)
#        point2 = np.zeros((4,))
#        point2[0] = points[frames[i+1]][0][0] * sx
#        point2[1] = points[frames[i+1]][0][1] * sy
#        point2[2:4] = (0, 1)
#        pointsGl2[:,c] = np.dot(T[fr2,:,:],point2)
        pointsGl = singlePointFeaturesTo3DPointsMatrix(points, sx, sy, idx=(fr1,fr2,))
        pointsGl1[:,c] = np.dot(T[fr1,:,:],pointsGl[0,:])
        pointsGl2[:,c] = np.dot(T[fr2,:,:],pointsGl[1,:])
        c += 1
    pointsGl1 = pointsGl1[0:3,:]
    pointsGl2 = pointsGl2[0:3,:]
    
    # Calculate DA
    dist = np.zeros((pointsGl1.shape[1],))
    for i in xrange(0,len(dist)):
        dist[i] = np.linalg.norm(pointsGl1[:,i]-pointsGl2[:,i])
    listDA = np.abs(dist - L)
    DA = stats.nanmean(listDA)
    return listDA, DA
    

def calculateRA(T, sx, sy, points, P):
    """Calculate Reconstruction Accuracy, as indicated in [Ref2]_. It needs
    1 single-point feature to be extracted for some US images of a calibration 
    quality assessment acquisition. These points (each for different US images)
    are reconstructed in global reference frame. RA is the mean of the norm of
    the difference between these points and the gold-standard points ``P``.
    
    Parameters
    ----------
    T : np.ndarray
        N x 4 x 4 array where ``T[i,:,:]`` represents the roto-translation matrix from US image reference frame to global reference frame, for time frame ``i``.
    
    sx, sy : float
        Number of mm for each pixel in US image, for horizontal and vertical axis (in *mm/pixel*).
    
    points : dict
        Dictionary where keys are frame numbers and values are lists of tuples, each one representing a point position in the corresponding US image. Only one tuple is needed.
    
    P : np.ndarray
        Gold-standard 3D positions (in *mm*) for reconstruction accuracy estimation.   
    
    Returns
    -------
    dist : np.ndarray
        Array containing distances, each on calculated by using a real 3D point and the reconstructed 3D point.
    
    DA : float
        Mean of ``dist`` ignoring nans.
        
    """

    # Calculate points in the global reference frame
    pointsGl1 = np.zeros((3,len(points))) * np.nan
    pointsGl2 = np.zeros((3,len(points))) * np.nan
    frames = np.sort(points.keys())
    for i in xrange(0, len(points)):
        if len(points[frames[i]]) <> 1:
#            pointsGl[:,i] = np.nan
            continue
        fr = frames[i]
        point = singlePointFeaturesTo3DPointsMatrix(points, sx, sy, idx=(fr,)).squeeze()
        pointsGl1[:,i] = np.dot(T[fr,:,:],point)[0:3]
        if fr < P.shape[0]:
            pointsGl2[:,i] = P[fr,:]
        else:
            pointsGl2[:,i] = np.nan
    #pointsGl1 = pointsGl1[0:3,:]
    
    # Calculate RA
    dist = np.zeros((pointsGl1.shape[1],))
    for i in xrange(0,pointsGl1.shape[1]): # np.linalg.norm has no 'axis' argument for NumPy 1.7.1
        dist[i] = np.linalg.norm(pointsGl1[:,i]-pointsGl2[:,i])
    RA = stats.nanmean(dist)
    return dist, RA
    

def calculateTimeDelayXCorr(s1, s2, s1Label, s2Label, timeVector, step, lagsBound=None, withPlots=True):
    """ Estimate the delay between two normalized signals by cross-correlation.
    Normalization consists of demeaning and dividing by the maximum of the rectified signal.
    From the cross-correlation signal, the maximum value within the time range 
    (``-lagsBound``, ``lagsBound``), in *s*, is found. The time instant in
    which that maximum occurs is the time delay estimation.
    If positive, ``s2`` is early with respect to ``s1``.
    
    Parameters
    ----------
    s1, s2 : np.ndarray
        Mono-dimensional arrays representing the signals to cross-correlate.
        
    s1Label, s2Label : str
        Strings for ``s1`` and ``s2`` to show in plots.
        
    timeVector : np.ndarray
        Time line (in *s*) for both original signals ``s1`` and ``s2``.
        It must contain the same number of frames as ``s1`` and ``s2``.
        
    step : float
        Resampling step for new time line for ``s1`` and ``s2``.
        The new time line goes from ``timeVector[0]`` to ``timeVector[-1]``.
        
    lagsBounds : mixed 
        Limiting range (in *s*) around which to search for the maximum cross-correlation value.
        If None, all the time line willbe used.
        
    withPlots : bool 
        If True, plots for results willbe shown (in blocking mode).
    
    Returns
    -------
    float
        Estimated time delay (in *s*).
        
    """    
    
    
    # Upsample signals
    x = np.arange(np.min(timeVector), np.max(timeVector), step=step)
    f1 = interp1d(timeVector, s1)
    a = f1(x)
    f2 = interp1d(timeVector, s2)
    b = f2(x)
    
    # Normalize signals
    a = a - np.mean(a)
    a = a / np.abs(np.max(a))
    b = b - np.mean(b)
    b = b / np.abs(np.max(b))
    
    # Cross-correlate signals
    dx = x[1] - x[0]
    lags, c, line, ax = plt.xcorr(a, b, normed=False, usevlines=False, maxlags=x.shape[0]-1)
    lags = lags * dx
    
    # Find maximum into the bounds
    if lagsBound == None:
        idx = (c == np.max(c))
    else:
        cc = c.copy()
        cc[np.abs(lags) > lagsBound] = 0.
        idx = (cc == np.max(cc))
    tau = lags[idx]
    
    # Plot some data if requested
    if withPlots:
        plt.subplot(3,1,1)
        plt.plot(x, a)
        plt.title(s1Label)
        plt.subplot(3,1,2)
        plt.plot(x, b)
        plt.title(s2Label)
        plt.subplot(3,1,3)
        plt.plot(lags, c)
        plt.hold(True)
        plt.plot([tau,tau],[0,c[idx]],'r')
        plt.plot([tau],[c[idx]],'or')
        plt.text(1.1*tau, 1.1*c[idx], 'delay = %.3f s' % tau)
        plt.title('Cross-correlation')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close(plt.gcf())
        
    return tau


def _printIterVariables(variables, init, values):
    
    for n, i, v in zip(variables, init, values):
        print '%s: %.5f (current), %.5f (init)' % (n, v, i)





    