## Inputs:
## V: n-by-3 numpy ndarray of vertex positions
## F: m-by-3 numpy ndarray of face indices
##
## Outputs:
## vertex areas: n-by-3 numpy ndarray

import numpy as np
import numpy.matlib as matlib
import scipy
import scipy.sparse 

def vertexAreas(V, F):
	vec1 = V[F[:,1],:] - V[F[:,0],:]
	vec2 = V[F[:,2],:] - V[F[:,0],:]
	faceNormal = np.cross(vec1, vec2) / 2
	faceArea = np.sqrt(np.power(faceNormal,2).sum(axis = 1))

	rowIdx = F.reshape(F.shape[0]*F.shape[1])
	colIdx = matlib.repmat(np.expand_dims(np.arange(F.shape[0]),axis=1),1,3).reshape(F.shape[0]*F.shape[1])
	data = np.ones([F.shape[0]*F.shape[1]]) / 3.0
	W = scipy.sparse.csr_matrix((data, (rowIdx, colIdx)), shape=(V.shape[0],F.shape[0]))
	return W*faceArea