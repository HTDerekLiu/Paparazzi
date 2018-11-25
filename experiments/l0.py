from paparazzi import Paparazzi
from imageL0Smooth import *


meshPath = '../assets/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../assets/bumpyCube_normalize_offset.obj' # offset surface of the geometry 

# L0 smoothing [Xu et al. 2011]
lam = 0.01
maxIter = 3000
imgSize = 256
windowSize = 0.5
lr = 1e-4
filterFunc = lambda img: imageL0Smooth(img,lam)
p = Paparazzi(meshPath,offsetPath,imgSize=imgSize,windowSize=windowSize)

p.run(maxIter,filterFunc)
