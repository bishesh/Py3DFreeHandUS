# -*- coding: utf-8 -*-
"""
.. module:: segment
   :synopsis: helper module for image segmentation

"""

import numpy as np
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Button, Slider, Lasso, RadioButtons
import tkFileDialog
import pickle
import cv2


def detectHoughLongestLine(I, thI=0.1, thCan1=50, thCan2=150, kerSizeCan=3, kerSizeDil=(3,3), thHou=100, minLineLength=2, maxLineGap=10):
    """Given a noisy gray-scale image containing 2 main blocks separated by a straight line, this function detects this line.
    The algorithm performs the following steps:
    
    1) Thresholding the input image
    2) Canny edge detector
    3) Dilation with a 1s-filled rectangular kernel
    4) Probabilistic Hough transform
    
    Parameters
    ----------
    I : np.ndarray(uint8)
        The input image.
        
    thI : float
        Threshold relative to the maximum representative gray level. 0 < thI < 1.
        
    thCan1 : int
        Canny edge detector lower threshold.
    
    thCan2 : int
        Canny edge detector higher threshold.
        
    kerSizeCan : int
        Sobel kernel size for Canny edge detector. Either 3, 5, 7.
        
    kerSizeDil : list
        Rectangular kernel size (height, width) for dilation.
        
    thHou : int
        Threshold for Hough transform output matrix.
        
    minLineLength : int
        Minimum line length for probabilistic Hough transform.   
        
    maxLineGap : int
        Maximum line gap for probabilistic Hough transform.
    
    Returns
    -------
    a, b : float
        Slope and intercept of the line. If no line is detected, these are np.nan.
    
    bw : np.ndarray
        Image as result of point 1.
    
    edges : np.ndarray
        Image as result of point 2.
    
    dilate : np.ndarray
        Image as result of point 3.
    
    """
    
    # Threshold image
    maxVal = np.iinfo(I.dtype).max
    th, bw = cv2.threshold(I,np.round(thI*maxVal),maxVal,cv2.THRESH_BINARY)
    # Detect edges
    edges = cv2.Canny(bw,thCan1,thCan2,apertureSize=kerSizeCan)
    # Dilate edges
    kernel = np.ones(kerSizeDil,I.dtype)
    dilate = cv2.dilate(edges, kernel, iterations=1)
    # Find longest line
    lines = cv2.HoughLinesP(dilate,1,np.pi/180,thHou,minLineLength,maxLineGap)
    maxL = 0
    if lines == None:
        a, b = np.nan, np.nan
    else:
        for x1,y1,x2,y2 in lines[0]:
            L = np.linalg.norm((x1-x2,y1-y2))
            if L > maxL:
                maxL = L
                a = float(y1 - y2) / (x1 - x2)
                b = y1 - a * x1
    # Return data
    return a, b, bw, edges, dilate


# inspired by: http://matplotlib.org/examples/widgets/buttons.html

class SegmentUI():
    """*(deprecated)* Class for performing manual point feature extraction.

    Parameters
    ----------
    I : np.array
        Nf x Nr x Nc (frames number x image row size x image column size) array containing grey levels data.
        
    data : dict
        Dictionary when keys are frame values and values are list of tuples. Each tuple represents coordinates for a single point in the image.
    
    Nclicks : int
        Number of point features to be extracted.
        
    block : bool
        If to block the window or not.
        In interactive Python shell mode, if True, it shows the main window and disables buttons usage. These have to be called
        manually from command line. If False, program flow is interrupted until the main window is closed, and buttons usage is enabled.
        In script mode, if True, it has the same behaviour as interactive shell. In this mondality, False value has no meaning since no
        window is shown.
            
    title : str
        Window title

    """
    
    def __init__(self, I, data={}, Nclicks=1, block=True, title=''):
        
        # Set sttributes
        self.I = I
        self.J, self.M, self.N = I.shape
        self.data = data
        self.Nclicks = Nclicks
        self.ind = 0
        
        # Create plot area
        self.fig, self.ax = plt.subplots()
        
        # Crete current idx text
        axidx = plt.axes([0.5, 0.15, 0.05, 0.065])
        axidx.set_axis_off()
        self.txt = plt.text(0, 0.5, '', axes=axidx)
        
        # Show first image
        self._showImage()
        self._showPoints()
        self._showCurrentIdx()
        
        # Define buttons
        axclick = plt.axes([0.09, 0.15, 0.1, 0.075])
        axreset = plt.axes([0.20, 0.15, 0.1, 0.075])
        axfileload = plt.axes([0.09, 0.05, 0.1, 0.075])
        axfilesave = plt.axes([0.20, 0.05, 0.1, 0.075])
        axprev = plt.axes([0.7, 0.15, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.15, 0.1, 0.075])
        bclick = Button(axclick, 'Click')
        bclick.on_clicked(self.click)
        breset = Button(axreset, 'Reset')
        breset.on_clicked(self.reset)
        bfileload = Button(axfileload, 'Load...')
        bfileload.on_clicked(self.fileLoad)
        bfilesave = Button(axfilesave, 'Save...')
        bfilesave.on_clicked(self.fileSave)
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev)
        
        # Plot title
        plt.fig.suptitle(title)
        
        # Adjust borders
        plt.subplots_adjust(bottom=0.3)    
        
        # Plot
        plt.show(block=block)
        
    
    def _showImage(self):
        
        plt.sca(self.ax)
        plt.cla()
        plt.imshow(self.I[self.ind,:,:].squeeze(), cmap=plt.cm.binary)
        plt.xlim(0, self.N)
        plt.ylim(self.M, 0)
        
        
    def next(self, event):
        """Show next image.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Calculate index
        self.ind += 1
        self.ind = self.ind % self.J
        self._showCurrentIdx()
        # Replot image
        self._showImage()
        # Plot points
        self._showPoints()
        

    def prev(self, event):
        """Show previous image.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Calculate index
        self.ind -= 1
        self.ind = self.ind % self.J
        self._showCurrentIdx()
        # Replot image
        self._showImage()
        # Plot points
        self._showPoints()
        
        
    def click(self, event): 
        """Allow to click on ``Nclicks`` manually in the current image
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """

        pts = plt.ginput(self.Nclicks)
        self.data[self.ind] = pts
        self._showPoints()
        
    
    def reset(self, event):
        """Delete the number of clicked points for the current image.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        del self.data[self.ind]
        self._clearPoints()
        
    
    def _showPoints(self):
        
        if self.ind in self.data:
            pts = self.data[self.ind]
            for i in xrange(0, len(pts)):
                plt.sca(self.ax)
                plt.plot(pts[i][0], pts[i][1], 'ro')
        plt.xlim(0, self.N)
        plt.ylim(self.M, 0)
            
        
    def _clearPoints(self):
        
        plt.sca(self.ax)
        self._showImage()
        
    
    def _showCurrentIdx(self):
        
        self.txt.set_text(str(self.ind+1) + '/' + str(self.J))
        
    
    def getData(self):
        """Get clicked points data.
        
        Returns
        -------
        dict
            For the format, see ``data`` in the constructor method.

        """
        
        return self.data
    
    
    def fileLoad(self, event):
        """Allow to load points data from file, by a user dialog.
        
        Parameters
        ----------
        event : param matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Select path
        filePath = tkFileDialog.askopenfilename()
        if len(filePath) == 0:
            return
        # Load file
        with open(filePath, "rb") as f:
            self.data = pickle.load(f)
        # Plot points
        self._showPoints()


    def fileSave(self, event):
        """Allow to save points data to file, by a user dialog.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Select path
        filePath = tkFileDialog.asksaveasfilename()
        if len(filePath) == 0:
            return        
        # Save file
        with open(filePath, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        











class ViewerUI(object):
    """Class for visualization of 2D image frames.

    Parameters
    ----------
    I : np.array
        Nf x Nr x Nc (frames number x image row size x image column size) array containing grey levels data.
        
    title : str
        Window title

    """
    
    def __init__(self, I, title=''):
        
        # Set sttributes
        self.I = I
        self.J, self.M, self.N = I.shape
        self.ind = 0
        
        # Create plot area
        self.fig, self.ax = plt.subplots()
        
        # Crete current idx text
        axidx = plt.axes([0.5, 0.15, 0.05, 0.065])
        axidx.set_axis_off()
        self.txt = plt.text(0, 0.5, '', axes=axidx)
        
        # Show first image
        self.update()
        
        # Define buttons
        axprev = plt.axes([0.7, 0.15, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.15, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev)
        
        # Define index slider
        axsliderInd = plt.axes([0.5, 0.10, 0.1, 0.065])
        self.sliderInd = Slider(axsliderInd, 'Frame', 1, self.J, valinit=1, valfmt='%0.0f')
        self.sliderInd.on_changed(self.changedInd)
        
        # Plot title
        self.fig.suptitle(title)
        
        # Adjust borders
        plt.subplots_adjust(bottom=0.3)    
        
        
    def update(self):
        """Show image and frame number.
        
        """
        self.showImage()
        self.showCurrentIdx()        
        
    
    def showImage(self):
        """Show image for current frame.
        """
        
        plt.sca(self.ax)
        plt.cla()
        plt.imshow(self.I[self.ind,:,:], cmap=plt.cm.gray)
        plt.xlim(0, self.N)
        plt.ylim(self.M, 0)
        
        
    def next(self, event):
        """Show next image.
        
        Parameters
        ----------
        event : param matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Calculate index
        self.ind += 1
        self.ind = self.ind % self.J
        self.update()
        

    def prev(self, event):
        """Show previous image.
        
        Parameters
        ----------
        event : param matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Calculate index
        self.ind -= 1
        self.ind = self.ind % self.J
        self.update()
        

    def changedInd(self, val):
        """Callback for frame index slider.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        
        # Set index
        valIn = val
        val = int(val)
        if val <> valIn:
            self.sliderInd.set_val(val)
        self.ind = val-1
        self.ind = self.ind % self.J
        self.update()
        
    
    def showCurrentIdx(self):
        """Show current frame number.
        """
        
        self.txt.set_text(str(self.ind+1) + '/' + str(self.J))
        
        
    def showViewer(self):
        """Show viewer.
        """
        plt.show()
        
        
    def closeViewer(self):
        """Close viewer.
        """
        plt.close(self.fig)        



class ViewerWithFeaturesUI(ViewerUI):
    """Class for visualization of 2D image frames and image features.
    
    Parameters
    ----------
    *args
        See ``ViewerUI.__init__()``.
            
    **kwargs
        See ``ViewerUI.__init__()``.
            
    """
    
    def __init__(self, *args, **kwargs):
        super(ViewerWithFeaturesUI, self).__init__(*args, **kwargs)
        self.featuresUI = None
    
    def next(self, *args):
        """Show next image and features. See method ``ViewerUI.next()``.
        """
        super(ViewerWithFeaturesUI, self).next(*args)
        self.featuresUI.updateData()
        self.featuresUI.showData()
        
    def prev(self, *args):
        """Show previous image and features. See class ``ViewerUI.prev()``.
        """
        super(ViewerWithFeaturesUI, self).prev(*args)
        self.featuresUI.updateData()
        self.featuresUI.showData()
        
    def changedInd(self, *args):
        """Callback for frame index slider. See class ``ViewerUI.changedInd()``.
        """
        super(ViewerWithFeaturesUI, self).changedInd(*args)
        self.featuresUI.updateData()
        self.featuresUI.showData()
        
    def getData(self):
        """Return points data.
        
        Returns
        -------
        dict
            Points data.

        """
        return self.featuresUI.getData()



class SegmentPointsUI(ViewerWithFeaturesUI):
    """Class for visualization of 2D image frames and manually segmentable points.
    
    Parameters
    ----------
    Npoints : int
        Number of point features per image to be extracted. 
        
    data : dict
        Dictionary when keys a frame values and values are list of tuples. Each tuple represents coordinates for a single point in the image.
    
    *args
        See ``ViewerWithFeaturesUI.__init__()``.
        
    **kwargs: 
        - 'title': window title.
    
    """

    def __init__(self, Npoints, data, *args, **kwargs):
        
        super(SegmentPointsUI, self).__init__(*args, title=kwargs['title'])
        del kwargs['title']
        
        self.featuresUI = OptsPointsUI(self, Npoints, data)
        


class MaskImageUI(ViewerWithFeaturesUI):
    """Class for visualization of 2D image frames and manually create a mask.
    
    Parameters
    ----------
    maskParams : int
        masking parameters. See ``OptsMaskImageUI.__init__()``
        
    data : dict
        Dictionary when keys a frame values and values are 2D binary Numpy matrices representing masks.
        
    *args :
        See ``ViewerWithFeaturesUI.__init__()``.
        
    **kwargs : 
        - 'title': window title.
    
    """

    def __init__(self, maskParams, data, *args, **kwargs):
        
        super(MaskImageUI, self).__init__(*args, title=kwargs['title'])
        del kwargs['title']
        
        self.featuresUI = OptsMaskImageUI(self, maskParams, data)
        
        
        
        
class SegmentPointsHoughUI(ViewerWithFeaturesUI):
    """Class for visualization of 2D image frames and automatically segmentable points lying on a line.
    The images are supposed to have two areas of diffrent grays levels, divided by a single line.
    For the details on the automatic line detection algorithm, see function ``detectHoughLongestLine()``.
    Automatically detected points can be manually adjusted.
    
    Parameters
    ----------
    Npoints : int
        Number of point features per image to be extracted.
        
    autoSegParams : dict
        Dictionary where keys are parameter names for function ``detectHoughLongestLine()``.
        
    dataConstr : list
        List of constraints for each point. Each element is a dictionary that can contain the follwing fields:
        - 'xPct': this imposes the x coordinate of the point to be a perecentage of the image width.    
    
    data : dict
        Dictionary when keys a frame values and values are list of tuples. Each tuple represents coordinates for a single point in the image.
    
    *args:
        See ``ViewerWithFeaturesUI.__init__()``.
        
    **kwargs: 
        - 'title': window title.
        - 'saveDataPath': existing folder path where to save automatically segmented images.

    """
    
    def __init__(self, Npoints, autoSegParams, dataConstr, data, *args, **kwargs):
        
        super(SegmentPointsHoughUI, self).__init__(*args, title=kwargs['title'])
        del kwargs['title']
        
        self.featuresUI = OptsPointsHoughUI(autoSegParams, dataConstr, self, Npoints, data, **kwargs)   
            
            
            
            
            
class OptsPointsUI(object):
    """Class adding manual points extraction capabilities to class ``ViewerUI`` or a derivate.
    
    Parameters
    ----------
    viewer : ViewerUI
        Instance of class ``ViewerUI`` or a derivate.
        
    Npoints : int
        Number of point features per image to be extracted. 
        
    data : dict
        Dictionary when keys a frame values and values are list of tuples. Each tuple represents coordinates for a single point in the image.
    
    """
            
    def __init__(self, viewer, Npoints, data):
        
        self.viewer = viewer
        self.Npoints = Npoints
        self.data = data
        self.pointsh = []
        
        plt.sca(self.viewer.ax)
        
        axclick = plt.axes([0.09, 0.15, 0.1, 0.075])
        self.bclick = Button(axclick, 'Click')
        self.bclick.on_clicked(self.click)        
        
        axreset = plt.axes([0.20, 0.15, 0.1, 0.075])
        self.breset = Button(axreset, 'Reset')
        self.breset.on_clicked(self.reset)
        
        axfileload = plt.axes([0.09, 0.05, 0.1, 0.075])
        self.bfileload = Button(axfileload, 'Load...')
        self.bfileload.on_clicked(self.fileLoad)
        
        axfilesave = plt.axes([0.20, 0.05, 0.1, 0.075])
        self.bfilesave = Button(axfilesave, 'Save...')
        self.bfilesave.on_clicked(self.fileSave)
        
        self.showPoints()

        

    def click(self, event): 
        """Allow to click on ``Npoints`` points manually in the current image
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """

        self.reset(event)
        plt.sca(self.viewer.ax)
        pts = plt.ginput(self.Npoints)
        self.data[self.viewer.ind] = pts
        self.showPoints()
        
    
    def reset(self, event):
        """Delete the number of clicked points for the current image.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        if self.viewer.ind in self.data:
            del self.data[self.viewer.ind]
        self.clearPoints()
        
    
    def showPoints(self):
        """Show points on screen for current frame.
        """
       
        plt.sca(self.viewer.ax)
        if self.viewer.ind in self.data:
            pts = self.data[self.viewer.ind]
            for i in xrange(0, len(pts)):
                line, = plt.plot(pts[i][0], pts[i][1], 'ro')
                self.pointsh.append(line)
        plt.xlim(0, self.viewer.N)
        plt.ylim(self.viewer.M, 0)
        

    def clearPoints(self):
        """Remove points from screen for current frame.
        """
        
        if self.pointsh <> []:
            for pointh in self.pointsh:            
                if pointh in self.viewer.ax.lines:
                    pointh.remove()
        self.pointsh = []
        
    
    def updateData(self):
        pass
        
        
    def showData(self):
        """Refresh the screen with the points for current frame.
        """
        
        self.clearPoints()
        self.showPoints()
        
        
    def fileLoad(self, event):
        """Allow to load points data from file, by a user dialog.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Select path
        filePath = tkFileDialog.askopenfilename()
        if len(filePath) == 0:
            return
        # Load file
        self.data = readFeaturesFile(filePath)
        # Plot points
        self.showPoints()


    def fileSave(self, event):
        """Allow to save points data to file, by a user dialog.
        
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Select path
        filePath = tkFileDialog.asksaveasfilename()
        if len(filePath) == 0:
            return        
        # Save file
        with open(filePath, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            
    def getData(self):
        """Return points data.
        
        Returns
        -------
        dict
            Points data.

        """
        return self.data
        
        
        

class OptsMaskImageUI(object):
    """Class adding manual masking capabilities to class ``MaskImageUI`` or a derivate.
    
    Parameters
    ----------
    viewer : ViewerUI
        Instance of class ``ViewerUI`` or a derivate.
        
    maskParams : int
        Dictionary for mask addition/creation:
        - 'selectorType': string indicating mask selector type: 'pen', 'lasso'.
        - 'width': width, in pixels, for the selector.
        If the selector is a pen, this is the pen thickness.
        If the selector is a lasso, this is ignored.
        
    data : dict
        Dictionary when keys a frame values and values are 2D binary Numpy matrices representing masks.
    
    """
       
    # Some parts are inspired by: http://matplotlib.org/examples/event_handling/lasso_demo.html   
    def __init__(self, viewer, maskParams, data):
        
        self.viewer = viewer
        self.maskParams = maskParams
        self.data = data
        self.mode = 'add'
        self.maskh = None
        
        plt.sca(self.viewer.ax)
        
        # Add other ui components
        axsliderWidth = plt.axes([0.07, 0.9, 0.10, 0.03])
        self.sliderWidth = Slider(axsliderWidth, 'Size', 3, 41, valinit=self.maskParams['width'])
        self.sliderWidth.on_changed(self.changedWidth)
        
        axradioType = plt.axes([0.07, 0.75, 0.10, 0.10])
        if self.maskParams['selectorType'] == 'lasso':
            active = 0
        else:
            active = 1
        self.radioType = RadioButtons(axradioType, ('Lasso', 'Pen'), active=active)
        self.radioType.on_clicked(self.changedType)
        
        axmode = plt.axes([0.09, 0.15, 0.1, 0.075])
        if self.mode == 'add':
            self.bmode = Button(axmode, 'Add')
        else:
            self.bmode = Button(axmode, 'Remove')
        self.bmode.on_clicked(self.modeChange)        
        
        axreset = plt.axes([0.20, 0.15, 0.1, 0.075])
        self.breset = Button(axreset, 'Reset')
        self.breset.on_clicked(self.reset)
        
        axfileload = plt.axes([0.09, 0.05, 0.1, 0.075])
        self.bfileload = Button(axfileload, 'Load...')
        self.bfileload.on_clicked(self.fileLoad)
        
        axfilesave = plt.axes([0.20, 0.05, 0.1, 0.075])
        self.bfilesave = Button(axfilesave, 'Save...')
        self.bfilesave.on_clicked(self.fileSave)
        
        if self.maskParams['selectorType'] in ['pen','lasso']:
            self.canvas = self.viewer.fig.canvas
            self.canvas.mpl_connect('button_press_event', self._onPress)
        
        self.showMask()
        

    def _postLasso(self, verts):
        # Relsease and delele lasso
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso
        # Create empty mask
        M = self.viewer.M
        N = self.viewer.N
        mask = np.zeros((M, N), dtype=np.bool)
        # Upsample vertices
        vertsPath = Path(verts).interpolated(100)
        verts = vertsPath.vertices
        verts = np.array(verts)
        # Fill mask
        idxR = np.round(verts[:,1]).astype(np.int32)
        idxC = np.round(verts[:,0]).astype(np.int32)
        idxR[idxR < 0] = 0
        idxC[idxC < 0] = 0
        idxR[idxR >= M] = M - 1
        idxC[idxC >= N] = N - 1
        mask[idxR, idxC] = True
        if self.maskParams['selectorType'] == 'pen':
            # Make dilation on mask
            d = self.maskParams['width'] + 2
            structure = np.ones((d,d), dtype=np.bool)
            mask = binary_dilation(mask, structure=structure)
        elif self.maskParams['selectorType'] == 'lasso':
            xMin, xMax = idxC.min(), idxC.max()
            yMin, yMax = idxR.min(), idxR.max()
            x = np.arange(xMin, xMax+1)
            y = np.arange(yMin, yMax+1)
            Np = x.shape[0] * y.shape[0]
            xv, yv = np.meshgrid(x, y)
            xv = np.reshape(xv.ravel(), (Np,1))
            yv = np.reshape(yv.ravel(), (Np,1))
            points = np.concatenate((xv,yv), axis=1).astype(np.int32)
            ind = vertsPath.contains_points(points)
            mask[points[ind,1], points[ind,0]] = True
        # Add/remove mask from current one if existing
        if self.viewer.ind in self.data:
            origMask = self.data[self.viewer.ind]
        else:
            origMask = np.zeros((self.viewer.M, self.viewer.N), dtype=np.bool)
        if self.mode == 'add':
            mask = origMask | mask
        else:
            idx = np.nonzero(mask)
            origMask[idx] = False
            mask = origMask
        # Assign new mask
        self.data[self.viewer.ind] = mask 
        # Show mask
        self.showData()
        
        
    def _onPress(self, event):
        if self.canvas.widgetlock.locked():
            return
        # Check if lasso is dragged onto another axes
        if event.inaxes <> self.viewer.ax:
            return
        # Create lasso
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self._postLasso)
        if self.maskParams['selectorType'] == 'pen':
            self.lasso.line.set_linewidth(self.maskParams['width'])
        self.lasso.line.set_color((1,0,0))
        # Acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)
        
        
    def changedWidth(self, val):
        """Callback for width.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        self.maskParams['width'] = int(val)
        self.updateData()
        self.showData()
        
        
    def changedType(self, label):
        """Callback for selector type.
        
        Parameters
        ----------
        label : str
            Current radiobutton label.
            
        """
        if label == 'Lasso':
            self.maskParams['selectorType'] = 'lasso'
        else:
            self.maskParams['selectorType'] = 'pen'
        self.updateData()
        self.showData()

        
    def modeChange(self, event): 
        """Allow to toggle add or remove modality.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        if self.mode == 'add':
            self.mode = 'del'
            self.bmode.label.set_text('Remove')
        else:
            self.mode = 'add'
            self.bmode.label.set_text('Add')
            
    
    def reset(self, event):
        """Delete the whole mask for current frame.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent 
            Event thrown when clicking on connected button.
        
        """
        if self.viewer.ind in self.data:
            del self.data[self.viewer.ind]
        self.clearMask()
        
    
    def showMask(self):
        """Show mask on screen for current frame.
        """
        plt.sca(self.viewer.ax)
        if self.viewer.ind in self.data:
            maskRGBA = np.zeros((self.viewer.M,self.viewer.N,4))
            maskRGBA[:,:,0] = self.data[self.viewer.ind]
            maskRGBA[:,:,3] = self.data[self.viewer.ind] * 0.5 # semi-transparent
            self.maskh = plt.imshow(maskRGBA)
        plt.xlim(0, self.viewer.N)
        plt.ylim(self.viewer.M, 0)
        

    def clearMask(self):
        """Remove whoole mask from screen for current frame.
        """
        if self.maskh <> None:
            if self.maskh in self.viewer.ax.images:
                self.maskh.remove()
        self.maskh = None
        
    
    def updateData(self):
        pass
        
        
    def showData(self):
        """Refresh the screen with the mask for current frame.
        """
        
        self.clearMask()
        self.showMask()
        
        
    def fileLoad(self, event):
        """Allow to load masks data from file, by a user dialog.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Select path
        filePath = tkFileDialog.askopenfilename()
        if len(filePath) == 0:
            return
        # Load file
        with open(filePath, "rb") as f:
            self.data = pickle.load(f)
        # Plot mask
        self.showData()


    def fileSave(self, event):
        """Allow to save masks data to file, by a user dialog.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event thrown when clicking on connected button.
            
        """
        
        # Select path
        filePath = tkFileDialog.asksaveasfilename()
        if len(filePath) == 0:
            return        
        # Save file
        with open(filePath, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            
    def getData(self):
        """Return masks data.
        
        Returns
        -------
        dict
            Masks data.

        """
        return self.data
            
            
            
            
class OptsPointsHoughUI(OptsPointsUI):
    """Class adding automatic (Hough transform) line segmentation capabilities to class ``ViewerUI`` or a derivate.
    The images are supposed to have two areas of diffrent grays levels, divided by a single line.
    For the details on the automatic line detection algorithm, see function ``detectHoughLongestLine()``.
    
    Parameters
    ----------
    autoSegParams : dict
        Dictionary where keys are parameter names for function ``detectHoughLongestLine()``.
        
    dataConstr : list
        List of constraints for each point. Each element is a dictionary that can contain the follwing fields:
        - 'xPct': this imposes the x coordinate of the point to be a perecentage of the image width.    

    *args
        See ``OptsPointsUI.__init__()``.
        
    """
            
    def __init__(self, autoSegParams, dataConstr, *args, **kwargs):
        """
        """
        super(OptsPointsHoughUI, self).__init__(*args)
        
        # Set segmentation parameters
        self.autoSegParams = autoSegParams
        self.dataConstr = dataConstr
        self.a = np.nan
        self.b = np.nan
        self.lineh = None
        self.p = None
        self.saveDataPath = ''
        if 'saveDataPath' in kwargs:
            self.saveDataPath = kwargs['saveDataPath']
        
        # Add other ui components
        axslider1 = plt.axes([0.07, 0.9, 0.10, 0.03])
        self.sliderThI = Slider(axslider1, 'Th', 0, 1, valinit=autoSegParams['thI'])
        self.sliderThI.on_changed(self.changedThI)
        
        axslider2 = plt.axes([0.07, 0.85, 0.10, 0.03])
        self.sliderThCan1 = Slider(axslider2, 'ThC1', 0, 300, valinit=autoSegParams['thCan1'], valfmt='%0.0f')
        self.sliderThCan1.on_changed(self.changedThCan1)
        
        axslider3 = plt.axes([0.07, 0.80, 0.10, 0.03])
        self.sliderThCan2 = Slider(axslider3, 'ThC2', 0, 300, valinit=autoSegParams['thCan2'], valfmt='%0.0f')
        self.sliderThCan2.on_changed(self.changedThCan2)
        
        axslider4 = plt.axes([0.07, 0.75, 0.10, 0.03])
        self.sliderKerSizeCan = Slider(axslider4, 'KerC', 3, 7, valinit=autoSegParams['kerSizeCan'], valfmt='%0.0f')
        self.sliderKerSizeCan.on_changed(self.changedKerSizeCan)
        
        axslider5 = plt.axes([0.07, 0.70, 0.10, 0.03])
        self.sliderKerSizeDilV = Slider(axslider5, 'KerDv', 3, 7, valinit=autoSegParams['kerSizeDil'][0], valfmt='%0.0f')
        self.sliderKerSizeDilV.on_changed(self.changedKerSizeDilV)
        
        axslider6 = plt.axes([0.07, 0.65, 0.10, 0.03])
        self.sliderKerSizeDilH = Slider(axslider6, 'KerDh', 3, 7, valinit=autoSegParams['kerSizeDil'][1], valfmt='%0.0f')
        self.sliderKerSizeDilH.on_changed(self.changedKerSizeDilH)
        
        axslider7 = plt.axes([0.07, 0.60, 0.10, 0.03])
        self.sliderThHou = Slider(axslider7, 'ThHou', 1, 1000, valinit=autoSegParams['thHou'], valfmt='%0.0f')
        self.sliderThHou.on_changed(self.changedThHou)
        
        axslider8 = plt.axes([0.07, 0.55, 0.10, 0.03])
        self.sliderMinLineLength = Slider(axslider8, 'minL', 1, 500, valinit=autoSegParams['minLineLength'], valfmt='%0.0f')
        self.sliderMinLineLength.on_changed(self.changedMinLineLength)
        
        axslider9 = plt.axes([0.07, 0.50, 0.10, 0.03])
        self.sliderMaxLineGap = Slider(axslider9, 'maxGap', 0, 100, valinit=autoSegParams['maxLineGap'], valfmt='%0.0f')
        self.sliderMaxLineGap.on_changed(self.changedMaxLineGap)
        
        axcreatepoints = plt.axes([0.01, 0.40, 0.25, 0.05])
        self.bcreatepoints = Button(axcreatepoints, 'Create points on line')
        self.bcreatepoints.on_clicked(self.autoCreatePoints)
        
        # Automatically detect points for all frames
        if self.data == {}:
            for i in range(0, self.viewer.J):
            #for i in range(0, 20):
                # Segment line
                print 'Segmenting line for line %d/%d' % (i+1,self.viewer.J)
                self.viewer.ind = i
                self.viewer.update()
                self.updateData()
                self.data[i] = self.p
                self.showData()
                # Save images to folder
                if self.saveDataPath <> '':
                    extent = self.viewer.ax.get_window_extent().transformed(self.viewer.fig.dpi_scale_trans.inverted())    # inspired by: http://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
                    figPath = self.saveDataPath + '/frame_%2d.png' % (i+1)      
                    self.viewer.fig.savefig(figPath, bbox_inches=extent.expanded(1., 1.))
        
        # Detect points for first frame
        self.viewer.ind = 0
        self.viewer.update()
        self.updateData()
        
        # Show data
        self.showData()
        
        
    def changedThI(self, val):
        """Callback for image threshold slider.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        self.autoSegParams['thI'] = val
        self.updateData()
        self.showData()
        
        
    def changedThCan1(self, val):
        """Callback for Canny edge detector lower threshold.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        self.autoSegParams['thCan1'] = val
        self.updateData()
        self.showData()
        
        
    def changedThCan2(self, val):
        """Callback for Canny edge detector higher threshold.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        self.autoSegParams['thCan2'] = val
        self.updateData()
        self.showData()
        

    def changedKerSizeCan(self, val):
        """Callback for Canny edge detector Sobel kernel size.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        valIn = val
        if val > 3 and val <= 3.5:
            val = 3    
        if val > 3.5 and val <= 5:
            val = 5
        elif val > 5 and val <= 7:
            val = 7
        if val <> valIn:
            self.sliderKerSizeCan.set_val(val)
        self.autoSegParams['kerSizeCan'] = val
        self.updateData()
        self.showData()
        
        
    def changedKerSizeDilV(self, val):
        """Callback for dilation kernel height.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        valIn = val
        if val > 3 and val <= 3.5:
            val = 3    
        if val > 3.5 and val <= 5:
            val = 5
        elif val > 5 and val <= 7:
            val = 7
        if val <> valIn:
            self.sliderKerSizeDilV.set_val(val)
        self.autoSegParams['kerSizeDil'][0] = val
        self.updateData()
        self.showData()
        
        
    def changedKerSizeDilH(self, val):
        """Callback for dilation kernel width.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        valIn = val
        if val > 3 and val <= 3.5:
            val = 3    
        if val > 3.5 and val <= 5:
            val = 5
        elif val > 5 and val <= 7:
            val = 7
        if val <> valIn:
            self.sliderKerSizeDilH.set_val(val)
        self.autoSegParams['kerSizeDil'][0] = val
        self.updateData()
        self.showData()
        

    def changedThHou(self, val):
        """Callback for probabilistic Hough transform threshold.
        
        Parameters
        ----------
        val : float
            Current slider value.
            
        """
        self.autoSegParams['thHou'] = int(val)
        self.updateData()
        self.showData()
        
        
    def changedMinLineLength(self, val):
        """Callback for probabilistic Hough transform minimum line length.
        
        Parameters
        ----------
        val : float
            Current slider value.
        
        """
        self.autoSegParams['minLineLength'] = int(val)
        self.updateData()
        self.showData()    
        

    def changedMaxLineGap(self, val):
        """Callback for probabilistic Hough transform maximum line gap.
        
        Parameters
        ----------
        val : float
            Current slider value.
        
        """
        self.autoSegParams['maxLineGap'] = int(val)
        self.updateData()
        self.showData()  
        
        
    def autoCreatePoints(self, event):
        """Create points on the automatically detected line.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent:
            Event thrown when clicking on connected button.
            
        """
        self.detectPoints()
        self.data[self.viewer.ind] = self.p
        self.showData()

        
    def showData(self):        
        """Refresh the screen with line the points for current frame.
        """
        self.clearLine()
        self.showLine()
        super(OptsPointsHoughUI, self).showData()
        
        
    def updateData(self):
        """Redetect automatically line and points.
        """
        self.detectLine()
        self.detectPoints()
        
        
    def detectPoints(self):
        """Detect points on the line using the constraints.
        """
        self.p = []
        for i in xrange(0, self.Npoints):
            constr = self.dataConstr[i]
            if 'xPct' in constr:
                x = constr['xPct'] * self.viewer.N
                y = self.a * x + self.b
                self.p.append((x, y))
        
        
    def detectLine(self):
        """Automatically detect the line.
        """
        self.a, self.b, bw, edges, dilate = detectHoughLongestLine(self.viewer.I[self.viewer.ind,:,:], **self.autoSegParams)
        
        
    def clearLine(self):
        """Remove line from screen for current frame.
        """
        if self.lineh <> None and self.lineh in self.viewer.ax.lines:
            self.lineh.remove()
        self.lineh = None
        
        
    def showLine(self):
        """Show line on screen for current frame.
        """
        x1 = 0
        y1 = self.a * x1 + self.b
        x2 = self.viewer.N
        y2 = self.a * x2 + self.b
        plt.sca(self.viewer.ax)
        self.lineh, = plt.plot((x1,x2),(y1,y2),'r-',linewidth=5)
        
        
        
        
        
def readFeaturesFile(filePath):
    """Read feature file data.
    
    Parameters
    ----------
    filePath : str
        Full file path for the features file.
    
    Returns
    -------
    dict
        dictionary when keys are frame values and values contain features data.

    """
    
    with open(filePath, "rb") as f:
        data = pickle.load(f)
    return data
    

def singlePointFeaturesTo3DPointsMatrix(fea, u, v, idx=None):
    """Transform a single points features structures to matrix containing 3D
    points data.
    
    Parameters
    ----------
    fea : dict
        Dictionary containing features data (see ``SegmentUI.__init__()``).
        
    u, v : float
        mm-to-pixel conversion factors (in *mm/pixel*) for horizontal and vertical coordinates.
        
    idx : list
        List of image frames number to be used. If None, all the available frames will be used.
    
    Returns
    -------
    np.ndarray
        Np x 4 matrix, where Np is the number of pointt features used 

    """
    
    if idx == None:
        idx = sorted(fea.keys())
    P = np.array([[fea[i][0][0]*u, fea[i][0][1]*v, 0., 1.] for i in idx])   # Np x 4
    return P

            
        
        
#if __name__ == '__main__':
#    
#    from image_utils import *
#    D, ds = readDICOM('C:/Users/Davide Monari/Desktop/KULeuvenJob/PythonSoftware/Ultrasound/cam_phantom_dave/test5_cam_br_Mevis.dcm')
#    I = pixelData2grey(D)
#        
#    parSeg = {}
#    parSeg['thI'] = 0.1
#    parSeg['thCan1'] = 50
#    parSeg['thCan2'] = 150
#    parSeg['kerSizeCan'] = 5
#    parSeg['kerSizeDil'] = [3, 3]
#    parSeg['thHou'] = 100
#    parSeg['minLineLength'] = 10
#    parSeg['maxLineGap'] = 10
#    
#    data = {}
#    dataConstr = [{'xPct':0.2},{'xPct':0.8}]    
#    
#    #ui = SegmentPointsHoughUI(2, parSeg, dataConstr, data, I, title='The title', saveDataPath='C:/Users/Davide Monari/Desktop/KULeuvenJob/PythonSoftware/Ultrasound/cam_phantom_dave/figs')
#    #ui = SegmentPointsUI(2, data, I, title='The title')  
#    
#    parMask = {}
#    parMask['selectorType'] = 'lasso'
#    parMask['width'] = 5
#    
#    ui = MaskImageUI(parMask, data, I, title='The title')
#    ui.showViewer()
        
        
        
        
        
        