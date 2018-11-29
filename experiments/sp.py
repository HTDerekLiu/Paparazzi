import os
import sys
sys.path.append('../utils')

from PaparazziFilter import *
from writeOBJ import *
import skimage
import skimage.segmentation

meshPath = '../meshes/bumpyCube_normalize.obj' # normalized geometry bounded by radius 1 cube
offsetPath = '../meshes/bumpyCube_normalize_offset.obj' # offset surface 

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
lr = 2e-5

def filterFunc(img):
    segs = skimage.segmentation.slic(img, compactness=compactness, n_segments=numSegments)
    return skimage.color.label2rgb(segs, img, kind='avg')

p = PaparazziFilter(meshPath,offsetPath,imgSize=imgSize,windowSize=windowSize)

V,F = p.run(maxIter,lr,filterFunc,outputFolder = outputFolder)
writeOBJ('sp.obj', V, F)
