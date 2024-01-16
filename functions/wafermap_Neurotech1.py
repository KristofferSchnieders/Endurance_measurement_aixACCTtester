# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:51:42 2024

@author: schnieders
"""
import numpy as np

#####################################################################
## Wafermaps
#####################################################################

def wafermap_Neurotech1_1R(): 
    '''
    Currently dummy

    Returns
    -------
    position : array like
        Positions of devices.
    name : array like
        Name of devices.
    geometry : array like
        Geometry. (For 1R probably not necessary)

    '''
    position, name, geometry = np.array([0]), np.array([0]), np.array([0])
    return position, name, geometry

def wafermap_Neurotech1_1T1R():
    '''
    Currently dummy

    Returns
    -------
    position : array like
        Positions of devices.
    name : array like
        Name of devices.
    geometry : array like
        Dimensions of transistors.

    '''
    position, name, geometry = np.array([0]), np.array([0]), np.array([0])
    return position, name, geometry
