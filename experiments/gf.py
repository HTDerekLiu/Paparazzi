from paparazzi import Paparazzi
from paparazzi.optimizer import NADAMOptimizer


meshPath = '../assets/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../assets/bumpyCube_normalize_offset.obj' # offset surface of the geometry 
# fast guided filter [He et al. 2015]

r = 6
eps = 0.02
maxIter = 3000
imgSize = 128
windowSize = 0.5
nadam_params = {"learning_rates":{0:1e-4,2500:1e-5}}


filterFunc = lambda img: guidedFilter(img, img, r, eps)

p = Paparazzi(filterFunc
        ,NADAMOptimizer
        ,nadam_params
        ,imgSize=imgSize
        ,windowSize=windowSize)

p.run(meshPath,offsetPath,maxIter)
