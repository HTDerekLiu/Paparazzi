import os
import sys
sys.path.append('../utils')

from PaparazziFilter import *
from writeOBJ import *
from imageL0Smooth import *

meshPath = '../assets/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../assets/bumpyCube_normalize_offset.obj' # offset surface 

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
lr = 5e-5

def filterFunc(img):
    return imageL0Smooth(img, lam)

p = PaparazziFilter(meshPath,offsetPath,imgSize=imgSize,windowSize=windowSize)
V,F = p.run(maxIter,lr,filterFunc, outputFolder = outputFolder)
writeOBJ('l0.obj', V, F)
