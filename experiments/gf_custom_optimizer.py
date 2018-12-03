import os
import sys


from paparazzi import *
from paparazzi.filters import *
from paparazzi.writeOBJ import *
from paparazzi.optimizer import Optimizer




class RK2Optimizer(Optimizer):
    def __init__(self
            ,gradFunc
            ,cleanupFunc = lambda i,x:(False,x)
            ,timestep=1e-2
            ,eps = 1e-8
            ):
        super(RK2Optimizer,self).__init__(gradFunc,cleanupFunc)
        self.timestep = timestep
        self.eps = 1e-8

    def __step__(self,V,mit):
        dV1 = self.gradFunc(V)
        tmpV = V - .5 * self.timestep * dV1
        dV = self.gradFunc(V)
        V[...] = V - self.timestep * dV
        return np.linalg.norm(V) < self.eps





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
opt_params = {"timestep":1e-4}

def filterFunc(img):
    return guidedFilter(img, img, r, eps_gf)

p = Paparazzi(filterFunc
        ,RK2Optimizer
        ,opt_params
        ,imgSize=imgSize
        ,windowSize=windowSize
        ,checkpoint_prefix=os.path.join(outputFolder,"gf_gdcheckpoint-")
        )
V,F = p.run(meshPath,offsetPath,maxIter)
writeOBJ('gf.obj', V, F)
