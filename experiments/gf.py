from paparazzi import Paparazzi


meshPath = '../assets/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../assets/bumpyCube_normalize_offset.obj' # offset surface of the geometry 
# fast guided filter [He et al. 2015]

r = 6
eps = 0.02
maxIter = 3000
imgSize = 128
windowSize = 0.5
lr = 1e-4
p = Paparazzi(meshPath,offsetPath,imgSize=imgSize,windowSize=windowSize)


filterFunc = lambda img: guidedFilter(img, img, r, eps)


p.run(maxIter,filterFunc)
