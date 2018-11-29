import os
import sys
sys.path.append('../utils')

from PaparazziFilter import *
from guidedFilter import *
from writeOBJ import *

meshPath = '../meshes/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../meshes/bumpyCube_normalize_offset.obj' # offset surface 

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
lr = 1e-4

def filterFunc(img):
    return guidedFilter(img, img, r, eps_gf)

p = PaparazziFilter(meshPath,offsetPath,imgSize=imgSize,windowSize=windowSize)
V,F = p.run(maxIter, lr, filterFunc, outputFolder = outputFolder)
writeOBJ('gf.obj', V, F)
