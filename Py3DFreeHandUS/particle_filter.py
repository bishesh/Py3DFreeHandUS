"""
.. module:: particle_filter
   :synopsis: particle filter implementation.


"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
from image_utils import histogramsSimilarity, NCC, createRandomInMaskCoords
import cv2
    
    
    
class ParticleFilter:
    """Particle filter implementation.
    
    These are the name conventions for variables:
   
    - nS: number of state variables.
    
    - nI: number of iterations performed.
    
    - nP: number of particles.
    
    - x: np.ndarray (nS x nP). Matrix containing data for each particle,
      for current iteration. Each row is a state variable.
    
    - xNext: np.ndarray (nS x nP). Matrix containing data for each particle, 
      for next iteration. Each row is a state variable.
    
    - xNextFromWeights: np.ndarray (nS). Vector containing state estimation
      for next iteration, only by using weights.
    
    - xNextEst: np.ndarray (nS). Vector containing state estimation for next
      iteration.
    
    - xEst: np.ndarray (nS x nI). Matrix containing history of estimated states, 
      in columns.
    
    - w: np.ndarray (nP). Vector of particle weights for current iteration.
    
    - wNext: np.ndarray (nP). Vector of particle weights for next iteration.
    
    - wNextNorm: np.ndarray (nP). wNext normalized on their sum.
    
    - addData: mixed. See method ``setAdditionalData()``.
    
    """
    
    def __init__(self):
        """Constructor
        """
        self.transModel = None
        self.observModel = None
        self.estimFromWeightsModel = None
        self.estimModel = None
        self.x = None
        self.xNext = None
        self.nParticles = None
        self.w = None
        self.xEst = None
        self.addData = None
        self.xNext = None
        self.wNext = None
        self.wNextNorm = None
        self.xNextFromWeights = None
        
    def setProbTransModel(self, model):
        """Set the probability/state transition model (i.e. p(x+1|x))
        
        Parameters
        ----------
        model : fun
            Function defining the transition model. The function must have
            these inputs: x, xEst (normally not necessary here), addData.
            It must return xNext.
            
        """
        self.transModel = model
    
    def setProbObservModel(self, model):
        """Set the observation model (i.e. p(y|x))
        
        Parameters
        ----------
        model : fun
            Function defining the observation model. The function must have
            these inputs: xNext, xEst, w, addData.
            It must return wNext.
            
        """
        self.observModel = model
        
    def setStateEstimatorFromWeightsFun(self, model):
        """Set the model for state estimation from weights 
        (e.g. maximum likelyhood, expectation, etc.)
        
        Parameters
        ----------
        model : fun
            Function defining the state estimation model from weights. 
            The function must have these inputs: xNext, wNextNorm.
            It must return xNextFromWeights.
            
        """
        self.estimFromWeightsModel = model
        
    def setProbEstimModel(self, model):
        """Set the a-posteriori probability model (i.e. p(x|Y))
        
        Parameters
        ----------
        model : fun
            Function defining the observation model. The function must have
            these inputs: xEst, xNext, wNextNorm, xNextFromWeights, addData.
            It must return xNextEst.
            
        """
        self.estimModel = model
        
    def setParticles(self, x):
        """Set particles data for current iteration.
        
        Parameters
        ----------
        x : np.ndarray (nS x nP)
            See x.
            
        """
        self.x = x
        self.xNext = x.copy()
        self.nParticles = x.shape[1]
        
    def setParticleWeights(self, w):
        """Set particles weights for current interation.
        
        Parameters
        ----------
        w : np.ndarray (nP)
            See w.
            
        """
        self.w = w
    
    def setState(self, xEst):
        """Set estimated states history.
        
        Parameters
        ----------
        xEst : np.ndarray (nS x nI)
            See xEst.
            
        """
        self.xEst = xEst
        
    def setAdditionalData(self, data):
        """Set additional data, to be passed to and used by model functions.
        
        Parameters
        ----------
        data : mixed
            This can be any kind of data.
            
        """
        self.addData = data
        
    def predict(self):
        """Perform the prediction step.
        """
        self.xNext = self.transModel(self.x, self.xEst, self.addData)
        
    def getPredictedParticles(self):
        """Get predicted particles.

        Returns
        -------
        np.ndarray (nS x nP)
            See xNext.
    
        """
        return self.xNext
    
    def update(self):
        """Perform the update (and weights normalization) step.
        """
        self.wNext = self.observModel(self.xNext, self.xEst, self.w, self.addData)
        self.wNextNorm = self.wNext / np.sum(self.wNext)        
        
    def getUpdatedParticleWeights(self):
        """Get weights for updated particles.

        Returns
        -------
        np.ndarray (nP)
            See wNextNorm.
    
        """
        return self.wNextNorm
        
    def estimateStateFromWeights(self):
        """Estimate state for next iteration, from particle weights only.
        """
        self.xNextFromWeights = self.estimFromWeightsModel(self.xNext, self.wNextNorm)
        
    def getEstimatedStateFromWeights(self):
        """Get state for next iteration, from particle weights only.

        Returns
        -------
        np.ndarray (nS)
            See xNextFromWeights.
    
        """
        return self.xNextFromWeights
            
    def estimateNextState(self):
        """Estimate and return state for next iteration.

        Returns
        -------
        np.ndarray (nS)
            See xNextEst.
    
        """
        return self.estimModel(self.xEst, self.xNext, self.wNextNorm, self.xNextFromWeights, self.addData)
        
    def getEffectParticlesNumber(self):
        """Get the number of effective particles, calculated by the inverse
        of the sum of squared normalized weights.
        
        Returns
        -------
        int
            Number of effective particles.
    
        """
        nEffParticles = int(1. / np.sum(self.wNextNorm**2))
        return nEffParticles
    
    def resample(self, mode='inverse_transform'):
        """Perform the resampling step.
        
        Parameters
        ----------
        mode : str
            Resampling method.
            If 'inverse_transform', the algorithm used is the inverse transform
            resampling.
        
        Returns
        -------
        np.ndarray (nS x nP)
            Matrix of the same size of x, but containing only particles with 
            heeavier weights.
            
        """
        pdf = self.wNextNorm
        data = np.arange(self.nParticles)
        if mode == 'inverse_transform':
            idx = inverseTransformSampling(data, pdf, self.nParticles)
        x = self.xNext[:,idx.round().astype(np.int32)]
        return x
        
        
def inverseTransformSampling(x, pdf, nSamples):
    """Perform data resampling based on inverse transform.
    
    Parameters
    ----------
    x : np.ndarray (nS x nP)
        Matrix where each column represents a sample.
        
    pdf : np.ndarray (nP)
        Probability for each sample.
        
    nSamples : int
        Number of samples to be resampled.
    
    Returns
    -------
    np.ndarray (nS x nP)
        Matrix of the same size of x, but containing only particles with 
        heavier weights.
                
    """
    # Inspired by http://www.nehalemlabs.net/prototype/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
    cumValues = np.cumsum(pdf)
    cumValues /= cumValues[-1] # can be removed later
    invCdf = interpolate.interp1d(cumValues, x, bounds_error=False, fill_value=0.)
    r = np.random.rand(nSamples)
    return invCdf(r)
            
        

def E(x, w):
    """Calculate Weighted average (expectation estimator) of samples.
    
    Parameters
    ----------
    x : np.ndarray (nS x nP)
        Matrix where each column represents a sample.
        
    w : np.ndarray (nP)
        Probability (weight) for each sample.
    
    Returns
    -------
    np.ndarray (nS)
        Estimated vector.    
    
    """
    xOut = np.average(x, axis=1, weights=w)
    return xOut
    
       
def ML(x, w):
    """Calculate maximum likelihood estimation from samples (i.e. the sample
    with the higher weight).
    
    Parameters
    ----------
    x : np.ndarray (nS x nP)
        Matrix where each column represents a sample.
        
    w : np.ndarray (nP)
        Probability (weight) for each sample.
    
    Returns
    -------
    np.ndarray (nS)
        Estimated vector.    
    
    """
    xOut = x[:,np.argmax(w)]
    return xOut
    

def arm01ConstVelFun(x, xEst, addData):
    """Implement first-oder autoregressive model (constant velocity) for a 
    2D point. State vector is composed by x and y position, followed by x and
    y velocity. Gaussian noise is added to both position and velocity.
    
    Parameters
    ----------
    x : np.ndarray (nS x nP)
        See x.
    
    xEst : np.ndarray (nS x nI)
        See xEst.
        
    addData : dict
        See addData.
        It must contain the key 'otherPars'containing the following keys:
        
        - 'sigmaPos': 2-elem vector containing standard deviation for the 
          position noise (x and y)
          
        - 'sigmaVel': 2-elem vector containing standard deviation for the 
          velocity noise (x and y)
        
    Returns
    -------
    np.ndarray (nS x nP)
        See xNext.
        
    """
    pars = addData['otherPars']
    sigmaPos = pars['sigmaPos'][:,None]
    sigmaVel = pars['sigmaVel'][:,None]
    nParticles = x.shape[1]
    dt = 1.
    noise = np.zeros((4,nParticles)) 
    noise[:2,:] = sigmaPos * np.random.randn(2,nParticles)
    noise[2:,:] = sigmaVel * np.random.randn(2,nParticles)
    xNext = np.empty_like(x)
    xNext[:2,:] = x[:2,:] + x[2:,:]*dt + noise[:2,:]
    xNext[2:,:] = x[2:,:] + noise[2:,:]
    return xNext
    
    
def arm01ConstVelEstFun(xEst, xFromWeights, pars):
    """Implement a-posteriori probability model, with adaptive state, for 
    first-oder autoregressive model (constant velocity) for a 2D point 
    (see ``arm01ConstVelFun()``). The estimated state is a weighted average 
    between xNextFromWeights and the last estimated state.

    Parameters
    ----------    
    xEst : np.ndarray (nS x nI)
        See xEst.
    
    xFromWeights : np.ndarray (nS).
        See xNextFromWeights.
        
    pars : dict
        Dictionary of parameters. It must contain the following keys:
        
        - 'alfaLearnPos': 2-elem vector containing weights for the position 
          part (x and y) of xNextFromWeights.
          
        - 'alfaLearnVel': 2-elem vector containing weights for the velocity 
          part (x and y) of xNextFromWeights.
        
    Returns
    -------
    np.ndarray (nS)
        See xNextEst.
        
    """
    # Adaptive state estimate equation for first-oder autoregressive model (constant velocity)
    xNextEst = np.zeros((4,))
    dt = 1.
    alfaLearnPos = pars['alfaLearnPos']
    alfaLearnVel = pars['alfaLearnVel']
    xNextEst[:2] = (1 - alfaLearnPos) * (xEst[:2,-1] + xEst[2:,-1]*dt) + alfaLearnPos * xFromWeights[:2]
    xNextEst[2:] = (1 - alfaLearnVel) * xEst[2:,-1] + alfaLearnVel * (xNextEst[:2] - xEst[:2,-1]) / dt
    return xNextEst
    

def bhattDistObsFun(xNext, xEst, w, addData):
    """Implement a Bhattacharyya distance-based observation model for image
    tracking.
    The probability distribution is modelled as a Gaussian distribution of the
    Bhattacharyya histogram distance between a target patch and patches centered 
    around the positional part of particles.
    The first 2 state variables must be x and y position of the patch center
    under tracking.

    Parameters
    ----------    
    xNext : np.ndarray (nS x nP)
        See xNext.    
    
    xEst : np.ndarray (nS x nI)
        See xEst.
    
    w : np.ndarray (nS).
        See w.
        
    addData : dict
        See addData.
        It must contain the key 'otherPars' containing the following keys:
        
        - 'nBins': number of histogram bins for Bhattacharyya distance calculation.
        
        - 'sigmaBhatta': sigma for Gaussian distance distribution.
        
        It must contain the following keys:
        
        - 'I': current image (only needed if posManuallySet is True or nI is 1)
        
        - 'boxSize': tuple containing image size (width, height)
        
        - 'posManuallySet': flag indicating if the target position is manually
          set for the current iteration.
        
    Returns
    -------
    np.ndarray (nS)
        See wNext.
        
    """
    # Classical Bhattacharyya distance-based observation model
    nParticles = xNext.shape[1]
    B = np.zeros((nParticles,))
    # Get additional data
    I = addData['I']      # current image
    w, h = addData['boxSize']   # box size
    posManuallySet = addData['posManuallySet']
    otherPars = addData['otherPars']
    nBins = otherPars['nBins']
    sigmaBhatta = otherPars['sigmaBhatta']
    # Get/calculate target histogram
    if posManuallySet or xEst.shape[1] == 1:
        c = xEst[:2,-1]
        T = I[c[1]-(h-1)/2:c[1]+(h-1)/2, c[0]-(w-1)/2:c[0]+(w-1)/2]
        Thist = cv2.calcHist([T],[0],None,[nBins],[0,256])
        addData['Thist'] = Thist
    else:
        Thist = addData['Thist']
    # Calculate particle weights
    for i in xrange(nParticles):
        c = xNext[:2,i]
        O = I[c[1]-(h-1)/2:c[1]+(h-1)/2, c[0]-(w-1)/2:c[0]+(w-1)/2]
        Ohist = cv2.calcHist([O],[0],None,[nBins],[0,256])    # histogram around particle 'i'
        C = histogramsSimilarity(Thist, Ohist, meas='bhattacharyya_coef')
        B[i] = (1 - C)**0.5
    wNext = np.exp(- B**2 / (2 * sigmaBhatta**2))
    return wNext
    
    
def RNCCObsFun(xNext, xEst, w, addData):
    """Implement a rectified NCC-based observation model for image tracking. 
    Rectified NCC (RNCC) is the same as NCC, but truncated to 0 when NCC is 
    lower than 0.
    The probability distribution is directly expressed as the RNCC value
    (0 <= RNCC <= 1) between a target patch and patches centered around the 
    positional part of particles.
    The first 2 state variables must be x and y position of the patch center
    under tracking.

    Parameters
    ----------    
    xNext : np.ndarray (nS x nP)
        See xNext.    
    
    xEst : np.ndarray (nS x nI)
        See xEst.
    
    w : np.ndarray (nS).
        See w.
        
    addData : dict
        See addData.
        It may contain the key 'otherPars' containing the following keys:
        
        - 'verbose': if True, more data is shown during computation.
            
        It must contain the following keys:
        
        - 'I': current image (only needed if posManuallySet is True or nI is 1)
            
        - 'boxSize': tuple containing image size (width, height)
            
        - 'posManuallySet': flag indicating if the target position is manually
          set for the current iteration.
            
        
    Returns
    -------
    np.ndarray (nS)
        See wNext.
        
    """
    # Classical Bhattacharyya distance-based observation model
    nParticles = xNext.shape[1]
    wNext = np.zeros((nParticles,))
    # Get additional data
    I = addData['I']      # current image
    w, h = addData['boxSize']   # box size
    posManuallySet = addData['posManuallySet']
    if 'otherPars' in addData:
        otherPars = addData['otherPars']
    else:
        otherPars = {}
    if 'targetMask' in otherPars:
        mask = otherPars['targetMask']
    else:
        mask = np.ones((h-1,w-1), dtype=np.bool)
    if 'obsFunVerbose' in otherPars:
        verbose = otherPars['obsFunVerbose']
    else:
        verbose = False  
    # Get/calculate target histogram
    if posManuallySet or xEst.shape[1] == 1:
        c = xEst[:2,-1]
        T = I[c[1]-(h-1)/2:c[1]+(h-1)/2, c[0]-(w-1)/2:c[0]+(w-1)/2]
        addData['T'] = T
    else:
        T = addData['T']
    #plt.imshow(T)
    #plt.show()
    # Calculate particle weights
    for i in xrange(nParticles):
        c = xNext[:2,i]
        O = I[c[1]-(h-1)/2:c[1]+(h-1)/2, c[0]-(w-1)/2:c[0]+(w-1)/2]
        if np.array_equal(O.shape, T.shape):
            wNext[i] = NCC(O[mask==True],T[mask==True])
            if verbose:  
                # Plot most important particles
                if wNext[i] > 0.9:
                    plt.subplot(221)
                    plt.imshow(O*mask, cmap=plt.cm.gray, vmin=0, vmax=255)
                    plt.title('Particle %d: %f' % (i,wNext[i]))
                    plt.subplot(222)
                    plt.imshow(I, cmap=plt.cm.gray, vmin=0, vmax=255)
                    plt.hold(True)
                    plt.plot(c[0], c[1], 'o', mfc='none', mec='r', scalex=False, scaley=False)
                    plt.subplot(223)
                    plt.imshow(T*mask, cmap=plt.cm.gray, vmin=0, vmax=255)
                    plt.show()
        else:
            wNext[i] = 0
    wNext[wNext<0] = 0
    return wNext
    

def adaptiveHistEstFun(xEst, xNext, wNextNorm, xNextFromWeights, addData):
    """Use a-posteriori probability model described in ``arm01ConstVelEstFun()``
    and perform target template histogram update with a certain learning rate, 
    for image tracking purpose. This function has to be used in combination with
    any histogram-distance-based observation model (e.g. ``bhattDistObsFun()``).
    The first 2 state variables must be x and y position of the patch center
    under tracking.

    Parameters
    ----------    
    xEst : np.ndarray (nS x nI)
        See xEst.
        
    xNext : np.ndarray (nS x nP)
        See xNext.
        
    wNextNorm : np.ndarray (nP)
        See wNextNorm.
    
    xNextFromWeights : np.ndarray (nS).
        See xNextFromWeights.
        
    addData : dict
        See addData.
        It must contain the key 'otherPars' containing the following keys:
        
        - 'alfaLearnHist': learning rate for target template.
        
        - keys required by ``arm01ConstVelEstFun()``.
        
        It must contain the following keys:
        
        - 'I': current image (only needed if posManuallySet is True)
        
        - 'boxSize': tuple containing image size (width, height)
        
        - 'posManuallySet': flag indicating if the target position is manually
          set for the current iteration.
        
    Returns
    -------
    np.ndarray (nS)
        See xNextEst.
        
    """
    # Get additional data
    I = addData['I']      # current image
    w, h = addData['boxSize']   # box size
    posManuallySet = addData['posManuallySet']
    otherPars = addData['otherPars']
    nBins = otherPars['nBins']
    alfaLearnHist = otherPars['alfaLearnHist']
    if posManuallySet:
        # Set non-positional state values
        pass
    else:
        # Update state
        xNextEst = arm01ConstVelEstFun(xEst, xNextFromWeights, otherPars)
        # Update target histogram
        Thist = addData['Thist']
        c = xNextFromWeights[:2]
        #c = xNextEst[:2]
        O = I[c[1]-(h-1)/2:c[1]+(h-1)/2, c[0]-(w-1)/2:c[0]+(w-1)/2]
        Ohist = cv2.calcHist([O],[0],None,[nBins],[0,256])
        Thist = (1 - alfaLearnHist) * Thist + alfaLearnHist * Ohist
        addData['Thist'] = Thist
    # Return estimated state
    return xNextEst
    
    
def adaptiveTemplateEstFun(xEst, xNext, wNextNorm, xNextFromWeights, addData):
    """Use a-posteriori probability model described in ``arm01ConstVelEstFun()``
    and perform target template update with a certain learning rate, for image 
    tracking purpose. This function has to be used in combination with any 
    template-match-based observation model (e.g. ``RNCCObsFun()``).
    The first 2 state variables must be x and y position of the patch center
    under tracking.

    Parameters
    ----------    
    xEst : np.ndarray (nS x nI)
        See xEst.
        
    xNext : np.ndarray (nS x nP)
        See xNext.
        
    wNextNorm : np.ndarray (nP)
        See wNextNorm.
    
    xNextFromWeights : np.ndarray (nS).
        See xNextFromWeights.
        
    addData : dict
        See addData.
        It must contain the key 'otherPars' containing the following keys:
        
        - 'alfaLearnTemplate': learning rate for target template.
        
        - keys required by ``arm01ConstVelEstFun()``.
        
        It must contain the following keys:
        
        - 'I': current image (only needed if posManuallySet is True)
        
        - 'boxSize': tuple containing image size (width, height)
        
        - 'posManuallySet': flag indicating if the target position is manually
          set for the current iteration.
        
    Returns
    -------
    np.ndarray (nS)
        See xNextEst.
        
    """
    # Get additional data
    I = addData['I']      # current image
    w, h = addData['boxSize']   # box size
    posManuallySet = addData['posManuallySet']
    otherPars = addData['otherPars']
    alfaLearnTemplate = otherPars['alfaLearnTemplate']
    if posManuallySet:
        # Set non-positional state values
        pass
    else:
        # Update state
        xNextEst = arm01ConstVelEstFun(xEst, xNextFromWeights, otherPars)
        # Update template histogram
        T = addData['T']
        c = xNextFromWeights[:2]
        #c = xNextEst[:2]
        O = I[c[1]-(h-1)/2:c[1]+(h-1)/2, c[0]-(w-1)/2:c[0]+(w-1)/2]
        T = (1 - alfaLearnTemplate) * T + alfaLearnTemplate * O
        addData['T'] = T
    # Return estimated state
    return xNextEst
    
    
def initRandParticlesInBoxFun(nParticles, xEst, addData):
    """Create particles inside a bounded box in case bounding box center is
    given manually or there are no estimated states yet.
    The first 2 state variables must be x and y position of the box point.

    Parameters
    ----------    
    nParticles : int
        Number of particles to create.
        
    xEst : np.ndarray (nS x nI)
        See xEst.
        
    addData : dict
        See addData.
        It must contain the following keys:
        
        - 'boxSize': tuple containing image size (width, height)
        
        - 'posManuallySet': flag indicating if the position position is manually
          set for the current iteration.

    Returns
    -------        
    xParticles : np.ndarray (nS x nP)
        See x. Velocity components are set to 0.
    
    """
    # Create random particles all contained inside a box
    w, h = addData['boxSize']
    posManuallySet = addData['posManuallySet']
    cx, cy = xEst[:2,-1]
    if posManuallySet or xEst.shape[1] == 1:
        xParticles = createRandomInMaskCoords(cx, cy, h, w, nParticles).T
        vel = np.zeros((2,nParticles))
        xParticles = np.vstack((xParticles, vel))
    return xParticles
    
    
    
    
    