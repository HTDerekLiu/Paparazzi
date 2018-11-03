## Inputs:
## X: a numpy array
##
## Outputs:
## X_normalized: row normalized numpy array

import numpy as np

def normalizeRow(X):
	l2Norm = np.sqrt((X * X).sum(axis=1))
	X_normalized = X / (l2Norm.reshape(X.shape[0],1)+1e-7)
	return X_normalized