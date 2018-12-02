from paparazzi import Paparazzi
from paparazzi.optimizer import NADAMOptimizer
from paparazzi.imageL0Smooth import *


meshPath = '../assets/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../assets/bumpyCube_normalize_offset.obj' # offset surface of the geometry 
# save results
outputFolder = './l0Results/'
try:
    os.stat(outputFolder)
except:
    os.mkdir(outputFolder)   

# L0 smoothing [Xu et al. 2011]
lam = 0.01
maxIter = 3000
imgSize = 256
windowSize = 0.5
nadam_params = {"learning_rates":{0:5e-5,2500:1e-5}}
filterFunc = lambda img: imageL0Smooth(img,lam)
p = Paparazzi(filterFunc
        ,NADAMOptimizer
        ,nadam_params
        ,imgSize=imgSize
        ,windowSize=windowSize
        ,checkpoint_prefix=os.path.join(outputFolder,"l0checkpoint-")
        )

V,F = p.run(meshPath,offsetPath,maxIter)
writeOBJ('l0.obj', V, F)
