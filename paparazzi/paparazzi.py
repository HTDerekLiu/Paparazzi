from readOBJ import *
from writeOBJ import *
from renderer import *
from eltopo import ElTopoMesh
from vertexNormals import vertexNormals
from vertexAreas import vertexAreas
from utils import computedNdV
#from A import A
import gc
import time
import skimage
import skimage.segmentation

import numpy as np
import scipy
import scipy.sparse as sparse


class Paparazzi(object):

    def __init__(self
            
            ,filterFunc
        # parameters for optimization (NADAM [Dozat et al. 2016])
        ,optimizer
        ,optimizer_params = {}
        ,imgSize=256
        ,windowSize = 0.5
        ,eltopoFreq = 30 # number of iterations to perform El Topo once
        , checkpoint_rate=30
        , checkpoint_prefix="checkpoint-"
            ):

        self.imgSize = imgSize
        self.filterFunc = filterFunc
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.elTopoFreq = eltopoFreq
        self.checkpoint_rate = 30
        self.checkpoint_prefix = checkpoint_prefix


        # initialize renderer
        # windowSize = 0.5

        self.windowSize = windowSize
        self.eltopoFreq = eltopoFreq

        self.renderer = PaparazziRenderer(imgSize = self.imgSize)
        
    def __del__(self):
        self.renderer.close()


    def __gradient__(self,V):
        
        self.__configure_renderer__(V)

        #compu9te partial derivatives
        dR = self.dR(V)
        dRdN = self.dRdN()
        dNdV = self.dNdV(V)
        
        # compute dV
        dV = dR * dRdN * dNdV
        # Transform into a |V|x3 vector
        dV = dV.reshape((dV.shape[1]/3, 3))
        return dV

    def __configure_renderer__(self,V):
        F = self.F
        # sample camera from offset surface
        viewIdx = np.searchsorted(self.VAoCumsum,np.random.rand())
        self.renderer.setCamera(self.windowSize, self.Vo[viewIdx,:], -self.VNo[viewIdx,:])

        # set light 
        x,y,z = self.renderer.getCameraFrame()
        self.renderer.setLights(x,y,z, np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])) # 3 rendererGB directional lights

        # set geometry
        self.renderer.setMesh(V,F)

    def __cleanup__(self, V,mit):
        # filter gradients eltopo
        did_cleanup = False
        if mit%self.elTopoFreq == 0:
            print("ElTopo cleanup starting...")
            t_eltopo = time.time()
            self.eltopo.update(V)
            V, newF = self.eltopo.getMesh()
            self.F = newF
            firstM = np.zeros(V.shape)
            secondM = np.zeros(V.shape)
            timeStep = 0
            print('eltopo finished:', time.time() - t_eltopo)
            gc.collect()
            did_cleanup=True

        if mit%self.checkpoint_rate == 0:
            filename = self.checkpoint_prefix + str(mit).zfill(5) + ".obj"
            print("Writing checkpoint obj: {}".format(filename))
            writeOBJ(filename,V,newF)


        return did_cleanup,V



    def run(self,meshPath,offsetPath,maxIter):


        self.V,self.F = readOBJ(meshPath)
        self.eltopo = ElTopoMesh(self.V,self.F)# build el topo mesh

        # read offset surface 
        self.Vo, self.Fo = readOBJ(offsetPath)
        self.VNo = vertexNormals(self.Vo,self.Fo)
        self.VAo = vertexAreas(self.Vo, self.Fo)
        self.VAoCumsum = np.cumsum(self.VAo) / self.VAo.sum() # for vertex area weighted uniform sampling


        self.renderer.setMesh(self.V,self.F)


        V = self.V
        opt = self.optimizer(self.__gradient__
                ,cleanupFunc=self.__cleanup__
                ,**self.optimizer_params)
        V = opt.run(V,maxIter)

        return V,F


    def dR(self,V):
        
        # rendering
        img = self.renderer.draw()
        
        filtImg = self.filterFunc(img)
        
        
        # compute delta R
        return (filtImg - img).reshape(1,3*self.imgSize**2)


    def dRdN(self):
        idxImg, actFIdx, actRow = self.renderer.draw("faceIdx")
        LDir, LC = self.renderer.getLights()
        row = np.concatenate((3*actRow, 3*actRow, 3*actRow,
            3*actRow+1, 3*actRow+1, 3*actRow+1,
            3*actRow+2, 3*actRow+2, 3*actRow+2))
        col = np.concatenate((3*actFIdx, 3*actFIdx+1, 3*actFIdx+2,
            3*actFIdx, 3*actFIdx+1, 3*actFIdx+2,
            3*actFIdx, 3*actFIdx+1, 3*actFIdx+2))
        my_ones = np.ones(len(actRow))
        data = np.concatenate((
            my_ones * np.dot(LDir[:,0], LC[:,0]), # dRdNx
            my_ones * np.dot(LDir[:,1], LC[:,0]), # dRdNy
            my_ones * np.dot(LDir[:,2], LC[:,0]), # dRdNz
            my_ones * np.dot(LDir[:,0], LC[:,1]), # dGdNx
            my_ones * np.dot(LDir[:,1], LC[:,1]), # dGdNy
            my_ones * np.dot(LDir[:,2], LC[:,1]), # dGdNz
            my_ones * np.dot(LDir[:,0], LC[:,2]), # dBdNx
            my_ones * np.dot(LDir[:,1], LC[:,2]), # dBdNy
            my_ones * np.dot(LDir[:,2], LC[:,2]), # dBdNz
            ))
        return sparse.csc_matrix((data, (row, col)), shape=(3*self.imgSize**2, 3*self.F.shape[0]))

    def dNdV(self,V):
        return computedNdV(V,self.F)
