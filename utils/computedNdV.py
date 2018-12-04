import numpy as np
import scipy 
import scipy.sparse as sparse

def computedNdV(V,F):
    row = np.zeros(F.shape[0]*27)
    col = np.zeros(F.shape[0]*27)
    data = np.zeros(F.shape[0]*27)
    idx = 0
    E = np.zeros((F.shape[0],3,3))
    E[:,:,0] = V[F[:,2],:] - V[F[:,1],:]
    E[:,:,1] = V[F[:,0],:] - V[F[:,2],:]
    E[:,:,2] = V[F[:,1],:] - V[F[:,0],:]
    FN_notunit = np.cross(E[:,:,0], E[:,:,1])
    lenFN = np.sqrt((FN_notunit**2).sum(axis=1))
    lenE = np.sqrt((E**2).sum(axis=1))
    n = FN_notunit / np.tile(lenFN[:,None], (1,3))
    minusndivAexpand = np.expand_dims(n / np.tile(-lenFN[:,None],(1,3)), axis = 1)
    rIdx = 3*np.array(range(F.shape[0]))
    for ii in xrange(3):

        b = E[:,:,(ii+0) % 3]
        bcrossn = np.cross(b,n)
        bcrossn = np.expand_dims(bcrossn, axis = 2)
        dndvVal = np.einsum('ijk,ikl->ijl',bcrossn,minusndivAexpand);

        # construct rIdx
        cIdx = 3*F[:, (ii  ) % 3]

        for jj in xrange(3):
            for kk in xrange(3):
                row[idx : idx+F.shape[0]] = rIdx + kk
                col[idx : idx+F.shape[0]] = cIdx + jj
                data[idx : idx+F.shape[0]] = dndvVal[:,kk,jj]
                idx += F.shape[0]
    dndv = sparse.coo_matrix((data, (row, col)), shape=(3*F.shape[0], 3*V.shape[0]))
    return dndv