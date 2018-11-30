import numpy as np
from vertexAreas import *

def normalizeShape(V,F):
    ## Normalize the shape to radius 1 cube
    ##
    ## Inputs:
    ## V: n-by-3 numpy ndarray of vertex positions
    ## F: m-by-3 numpy ndarray of face indices
    ##
    ## Outputs:
    ## Vout: n-by-3 numpy ndarray of vertex positions
    VA = vertexAreas(V,F)
    Vmean = np.sum(V * VA[:,None],axis = 0) / np.sum(VA)
    Vout = V.copy()
    Vout = Vout - Vmean
    Vout /= Vout.max()
    return Vout