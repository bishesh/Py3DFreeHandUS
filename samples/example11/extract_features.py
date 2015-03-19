# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 14:06:55 2014

@author: Francesco
"""

from Py3DFreeHandUS import process

c = process.Process()
c.setUSFiles(('dyn3_Mevis.dcm',))
c.extractFeatureFromUSImages(feature='1_point', segmentation='manual')