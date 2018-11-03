# compute edges from faces

import numpy as np

def edges(F):
	idx = np.array([[0,1], [1,2], [2,0]])
	EIdx1 = np.reshape(F[:,idx[:,0:1]], (np.product(F.shape),1))
	EIdx2 = np.reshape(F[:,idx[:,1:2]], (np.product(F.shape),1))
	EAll = np.concatenate((EIdx1, EIdx2), axis =1)

	temp = np.sort(EAll, axis = 1)
	mulNum = np.power(10, (len(str(F.shape[0]))))
	uniqueEID = temp[:,0] * mulNum + temp[:,1]
	sortEIdx = np.argsort(uniqueEID)
	EAll = EAll[sortEIdx,:] # sort the EAll from small to large (columwise)

	uniqueEIdx = np.where(np.sort(EAll, axis = 1)[:,0] == EAll[:,0])[0]
	edge = EAll[uniqueEIdx,:]

	## ===================
	##         RIP
	## ===================
	# idx = np.array([[0,1], [1,2], [2,0]])
	# edgeIdx1 = np.reshape(F[:,idx[:,0:1]], (np.product(F.shape),1))
	# edgeIdx2 = np.reshape(F[:,idx[:,1:2]], (np.product(F.shape),1))
	# edge = np.concatenate((edgeIdx1, edgeIdx2), axis =1)
	# edge = np.concatenate((edgeIdx1, edgeIdx2), axis =1)
	# edge = np.sort(edge, axis = 1)
	# edge = np.unique(edge, axis = 0)
	return edge