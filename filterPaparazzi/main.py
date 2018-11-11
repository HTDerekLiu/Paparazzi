from include import *
from PaparazziRenderer import *

meshPath = './bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = './bumpyCube_normalize_offset.obj' # offset surface of the geometry 
filterName = 'GF' # set the filter in use (options: "L0", "GF", "Quan", "SP")

if filterName == 'L0': # L0 smoothing [Xu et al. 2011]
    lam = 0.01
    maxIter = 3000
    imgSize = 256
    windowSize = 0.5
    lr = 1e-4
elif filterName == 'GF': # fast guided filter [He et al. 2015]
    r = 6
    eps = 0.02
    maxIter = 3000
    imgSize = 128
    windowSize = 0.5
    lr = 1e-4
elif filterName == 'SP': # SLIC superpixel [Achanta et al. 2012]
    compactness = 10
    numSegments = 100
    maxIter = 3000
    imgSize = 256
    windowSize = 0.3
    lr = 6e-5

# read mesh data
V,F = readOBJ(meshPath)
el = ElTopoMesh(V,F) # build el topo mesh

# read offset surface 
Vo, Fo = readOBJ(offsetPath)
VNo = vertexNormals(Vo,Fo)
VAo = vertexAreas(Vo, Fo)
VAoCumsum = np.cumsum(VAo) / VAo.sum() # for vertex area weighted uniform sampling

# initialize renderer
# windowSize = 0.5
R = PaparazziRenderer(imgSize = imgSize)
R.setMesh(V,F)

# parameters for optimization (NADAM [Dozat et al. 2016])
timeStep = 0
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
firstM = np.zeros(V.shape)
secondM = np.zeros(V.shape)
eltopoFreq = 30 # number of iterations to perform El Topo once

for iteration in xrange(maxIter):
    # reduce learning rate
    if iteration == 2500:
        lr = 1e-5

    # sample camera from offset surface
    viewIdx = np.searchsorted(VAoCumsum,np.random.rand())
    R.setCamera(windowSize, Vo[viewIdx,:], -VNo[viewIdx,:])

    # set light 
    x,y,z = R.getCameraFrame()
    R.setLights(x,y,z, np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])) # 3 RGB directional lights

    # set geometry
    R.setMesh(V,F)

    # rendering
    img = R.draw()

    # image filters 
    if filterName == 'L0':
        filtImg = imageL0Smooth(img, lam)
    elif filterName == 'GF':
        filtImg = guidedFilter(img, img, r, eps)
    elif filterName == 'SP':
        segs = skimage.segmentation.slic(img, compactness=compactness, n_segments=numSegments)
        filtImg = skimage.color.label2rgb(segs, img, kind='avg')

    # draw current image
    if iteration % 25 == 0:
        misc.imsave('curImg.jpg', (img*255).astype(np.uint8))
        misc.imsave('filtImg.jpg', (filtImg*255).astype(np.uint8))

    # compute delta R
    dR = (img - filtImg).reshape(1,3*imgSize**2)

    # compute dRdN
    idxImg, actFIdx, actRow = R.draw("faceIdx")
    LDir, LC = R.getLights()
    row = np.concatenate((3*actRow, 3*actRow, 3*actRow,
        3*actRow+1, 3*actRow+1, 3*actRow+1,
        3*actRow+2, 3*actRow+2, 3*actRow+2))
    col = np.concatenate((3*actFIdx, 3*actFIdx+1, 3*actFIdx+2,
        3*actFIdx, 3*actFIdx+1, 3*actFIdx+2,
        3*actFIdx, 3*actFIdx+1, 3*actFIdx+2))
    data = np.concatenate((
        np.ones(len(actRow)) * np.dot(LDir[:,0], LC[:,0]), # dRdNx
        np.ones(len(actRow)) * np.dot(LDir[:,1], LC[:,0]), # dRdNy
        np.ones(len(actRow)) * np.dot(LDir[:,2], LC[:,0]), # dRdNz
        np.ones(len(actRow)) * np.dot(LDir[:,0], LC[:,1]), # dGdNx
        np.ones(len(actRow)) * np.dot(LDir[:,1], LC[:,1]), # dGdNy
        np.ones(len(actRow)) * np.dot(LDir[:,2], LC[:,1]), # dGdNz
        np.ones(len(actRow)) * np.dot(LDir[:,0], LC[:,2]), # dBdNx
        np.ones(len(actRow)) * np.dot(LDir[:,1], LC[:,2]), # dBdNy
        np.ones(len(actRow)) * np.dot(LDir[:,2], LC[:,2]), # dBdNz
        ))
    dRdN = sparse.csc_matrix((data, (row, col)), shape=(3*imgSize**2, 3*F.shape[0]))

    # compute dNdV
    dNdV = computedNdV(V,F)

    # compute dV
    dV = dR * dRdN * dNdV
    dV = -dV.reshape((dV.shape[1]/3, 3))

    # perform NADAM update
    timeStep += 1
    firstM = beta1*firstM + (1-beta1)*dV
    secondM = beta2*secondM + (1-beta2)*(dV**2)
    firstM_cor = firstM / (1-np.power(beta1,timeStep))
    secondM_cor = secondM / (1-np.power(beta2,timeStep))
    newV = V - lr/(np.sqrt(secondM_cor)+eps) * (beta1*firstM_cor + ((1-beta1)*dV/(1-np.power(beta1,timeStep)))) 

    if (iteration+1) % 1 == 0:
        print "iteration: %d/%d" % (iteration+1, maxIter)

    # filter gradients eltopo
    if np.mod(iter, eltopoFreq) == 0:
        t_eltopo = time.time()
        el.update(newV)
        newV, newF = el.getMesh()
        firstM = np.zeros(newV.shape)
        secondM = np.zeros(newV.shape)
        F = newF
        timeStep = 0
        print 'eltopo time:', time.time() - t_eltopo
        gc.collect()
    V = newV

# el.update(newV)
# V,F = el.getMesh()
# writeOBJ('resultMesh_' + filterName + '.obj', V, F)
R.close()
