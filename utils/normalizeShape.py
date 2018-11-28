import numpy as np
from vertexAreas import *

def normalizeShape(V,F):
    VA = vertexAreas(V,F)
    Vmean = np.sum(V * VA[:,None],axis = 0) / np.sum(VA)
    V = V-Vmean
    V /= V.max()
    return V