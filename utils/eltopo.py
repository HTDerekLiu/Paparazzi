import numpy as np
import pyeltopo

class ElTopoMesh(object):
    def __init__(self, V,F):
        self.V = np.copy(V.astype(np.float64))
        self.F = np.copy(F.astype(np.int32))
        self.eltopo = pyeltopo.ElTopoTracker(self.V.T,self.F.T, True)
    def update(self, U, changeTopology = True):
        UT = U.astype(np.float64).T
        if changeTopology == True:
            self.eltopo.step(UT, 1)
        else:
            self.eltopo.integrate(UT, 1)
        self.V = self.eltopo.get_vertices().T
        self.F = self.eltopo.get_triangles().T
    def splitFace(self, FIdxList):
        for ii in xrange(len(FIdxList)):
            self.eltopo.split_triangle(ii)
        self.V = self.eltopo.get_vertices().T
        self.F = self.eltopo.get_triangles().T
    def getMesh(self):
        return self.V, self.F
