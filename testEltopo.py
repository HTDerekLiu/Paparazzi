import numpy as np
v = np.array([[0,0,0],[0,1,0],[1,0,0]],dtype=np.float64)
f  = np.array([[0,1,2]],dtype=np.int32)
print v.shape
print f.shape
import pyeltopo
e = pyeltopo.ElTopoTracker(v.T,f.T, True)
print(dir(e.init_params))
print(e.init_params.use_fraction)
e.init_params.use_fraction=False
print(e.init_params.use_fraction)