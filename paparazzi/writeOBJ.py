import numpy as np

def writeOBJ(filePath,V,F):
	## write .obj file
    ##
    ## Inputs:
	## filePath: path to the obj file
    ## V: n-by-3 numpy ndarray of vertex positions
    ## F: m-by-3 numpy ndarray of face indices
	f = open(filePath, 'w')
	for ii in range(V.shape[0]):
		string = 'v ' + str(V[ii,0]) + ' ' + str(V[ii,1]) + ' ' + str(V[ii,2]) + '\n'
		f.write(string)
	Ftemp = F + 1
	for ii in range(F.shape[0]):
		string = 'f ' + str(Ftemp[ii,0]) + ' ' + str(Ftemp[ii,1]) + ' ' + str(Ftemp[ii,2]) + '\n'
		f.write(string)
	f.close()