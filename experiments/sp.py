from paparazzi import Paparazzi
from paparazzi.optimizer import NADAMOptimizer
from paparazzi.imageL0Smooth import *
from writeOBJ import *
import skimage
import skimage.segmentation


meshPath = '../assets/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../assets/bumpyCube_normalize_offset.obj' # offset surface of the geometry 
# save results
outputFolder = './spResults/'
try:
    os.stat(outputFolder)
except:
    os.mkdir(outputFolder)   

# SLIC superpixel [Achanta et al. 2012]
compactness = 10
numSegments = 100
maxIter = 3000
imgSize = 256
windowSize = 0.3
nadam_params = {"learning_rates":{0:2e-5,2500:1e-5}}


def filterFunc(img):
    segs = skimage.segmentation.slic(img, compactness=compactness, n_segments=numSegments)
    return skimage.color.label2rgb(segs, img, kind='avg')

p = Paparazzi(filterFunc
        ,NADAMOptimizer
        ,nadam_params
        ,imgSize=imgSize
        ,windowSize=windowSize
        ,checkpoint_prefix=os.path.join(outputFolder,"spcheckpoint-")
        )


V,F = p.run(meshPath,offsetPath,maxIter)
writeOBJ('sp.obj', V, F)
