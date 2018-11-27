import sys
#this needs to change
#sys.path.append('/usr/local/libigl/python')  # change it according to python libigl location

# from utils



# spetial
import pyigl as igl
from iglhelpers import *


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

def normalizeShape(V,F):
    VA = vertexAreas(V,F)
    Vmean = np.sum(V * VA[:,None],axis = 0) / np.sum(VA)
    V = V-Vmean
    V /= V.max()
    return V

# guided filter
def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)


    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst

def gf(I, p, r, eps, s=None):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = scipy.ndimage.zoom(fullI, [1/s, 1/s, 1], order=1)
        p = scipy.ndimage.zoom(fullP, [1/s, 1/s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    mI_r = box(I[:,:,0], r) / N
    mI_g = box(I[:,:,1], r) / N
    mI_b = box(I[:,:,2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:,:,0]*p, r) / N
    mIp_g = box(I[:,:,1]*p, r) / N
    mIp_b = box(I[:,:,2]*p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:,:,0] * I[:,:,0], r) / N - mI_r * mI_r;
    var_I_rg = box(I[:,:,0] * I[:,:,1], r) / N - mI_r * mI_g;
    var_I_rb = box(I[:,:,0] * I[:,:,2], r) / N - mI_r * mI_b;

    var_I_gg = box(I[:,:,1] * I[:,:,1], r) / N - mI_g * mI_g;
    var_I_gb = box(I[:,:,1] * I[:,:,2], r) / N - mI_g * mI_b;

    var_I_bb = box(I[:,:,2] * I[:,:,2], r) / N - mI_b * mI_b;

    a = np.zeros((h, w, 3))
    for i in xrange(h):
        for j in xrange(w):
            sig = np.array([
                [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]],
                [var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j]],
                [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]]
            ])
            covIp = np.array([covIp_r[i,j], covIp_g[i,j], covIp_b[i,j]])
            a[i,j,:] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b

    meanA = box(a, r) / N[...,np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = scipy.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = scipy.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q

def guidedFilter(srcImg, guideImg, r, eps):
    # Fast guided filter
    outImg = np.zeros_like(srcImg)
    for ii in xrange(3): # guide each channel seperately
        outImg[:,:,ii] = gf(srcImg, guideImg[:,:,ii], r, eps)
    return outImg

# def closestPt(pt, pts):
#   dists = pts - pt
#   distsSquare = np.einsum('ij,ij->i', dists, dists)
#   return np.argmin(distsSquare)
