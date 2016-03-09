# -*- coding: utf-8 -*-
"""
.. module:: muscles_analysis
   :synopsis: module for analyzing image-based muscles properties

"""

from tracking import trackMTJ
import numpy as np

def MTJlengths(P1, P2, P3):
    """Calculate muscle-tendon junction (MJT) lengths based on insterion and MJT position.
    
    Parameters
    ----------
    P1 : np.ndarray
        3-elements array containing 3D coordinates for muscle insertion.
        
    P2 : np.ndarray
        3-elements array containing 3D coordinates for tendon insertion.
        
    P3 : np.ndarray
        3-elements array containing 3D coordinates for MJT.
    
    Returns
    -------
    dict
        Dictionary with the following keys:
        
        - 'Dmuscle': distance between P1 and P3
        - 'Dtendon': distance between P2 and P3
        - 'Dcomplex': distance between P1 and P2
        - 'DmusclePct': ratio between Dmuscle and Dmuscle + Dtendon
        - 'DtendonPct': ratio between Dtendon and Dmuscle + Dtendon
    
    """    
    
    res = {}
    res['Dmuscle'] = np.linalg.norm(P1 - P3)
    res['Dtendon'] = np.linalg.norm(P2 - P3)
    res['Dcomplex'] = np.linalg.norm(P1 - P2)
    res['DmusclePct'] = res['Dmuscle'] / (res['Dmuscle'] + res['Dtendon'])
    res['DtendonPct'] = res['Dtendon'] / (res['Dmuscle'] + res['Dtendon'])
    
    return res
    