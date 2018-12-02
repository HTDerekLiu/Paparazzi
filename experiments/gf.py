import os
import sys


from paparazzi import *
from paparazzi.filters import *
from paparazzi.writeOBJ import *
from paparazzi.optimizer import NADAMOptimizer

meshPath = '../assets/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../assets/bumpyCube_normalize_offset.obj' # offset surface of the geometry 
# fast guided filter [He et al. 2015]

# save results
outputFolder = './gfResults/'
try:
    os.stat(outputFolder)
except:
    os.mkdir(outputFolder)   

# fast guided filter [He et al. 2015]
r = 15
eps_gf = 1e-8
maxIter = 3000
imgSize = 128
windowSize = 0.5
nadam_params = {"learning_rates":{0:1e-4,2500:1e-5}}

def filterFunc(img):
    return guidedFilter(img, img, r, eps_gf)

p = Paparazzi(filterFunc
        ,NADAMOptimizer
        ,nadam_params
        ,imgSize=imgSize
        ,windowSize=windowSize
        ,checkpoint_prefix=os.path.join(outputFolder,"gfcheckpoint-")
        )
V,F = p.run(meshPath,offsetPath,maxIter)
writeOBJ('gf.obj', V, F)
