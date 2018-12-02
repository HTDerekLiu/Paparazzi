import numpy as np
import sys
import scipy
import scipy.sparse 
from normalizeRow import *
import numpy.matlib as matlib

def vertexNormals(V, F):    
	## Compute vertex normal
	##
	## Inputs:
	## V: n-by-3 numpy ndarray of vertex positions
	## F: m-by-3 numpy ndarray of face indices
	##
	## Outputs:
	## vertex normals: n-by-3 numpy ndarray
	vec1 = V[F[:,1],:] - V[F[:,0],:]
	vec2 = V[F[:,2],:] - V[F[:,0],:]
	FN = np.cross(vec1, vec2) / 2
	faceArea = np.sqrt(np.power(FN,2).sum(axis = 1))
	FN_normalized = normalizeRow(FN+sys.float_info.epsilon)

	VN = np.zeros(V.shape)
	rowIdx = F.reshape(F.shape[0]*F.shape[1])
	colIdx = matlib.repmat(np.expand_dims(np.arange(F.shape[0]),axis=1),1,3).reshape(F.shape[0]*F.shape[1])
	weightData = matlib.repmat(np.expand_dims(faceArea,axis=1),1,3).reshape(F.shape[0]*F.shape[1])
	W = scipy.sparse.csr_matrix((weightData, (rowIdx, colIdx)), shape=(V.shape[0],F.shape[0]))
	vertNormal = W*FN_normalized
	vertNormal = normalizeRow(vertNormal)
	return vertNormal