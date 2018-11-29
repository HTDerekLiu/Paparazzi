from readOBJ import *
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



class Paparazzi(object):

    def __init__(self
            
            ,filterFunc
        # parameters for optimization (NADAM [Dozat et al. 2016])
        ,optimizer
        ,optimizer_params = {}
        ,imgSize=256
        ,windowSize = 0.5
        ,eltopoFreq = 30 # number of iterations to perform El Topo once
            ):

        self.imgSize = imgSize
        self.filterFunc = filterFunc
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params



        # initialize renderer
        # windowSize = 0.5

        self.windowSize = windowSize
        self.eltopoFreq = eltopoFreq

        self.renderer = PaparazziRenderer(imgSize = self.imgSize)
        
    def __del__(self):
        self.renderer.close()

    def gradient(self,V):
        F = self.F
        # sample camera from offset surface
        viewIdx = np.searchsorted(self.VAoCumsum,np.random.rand())
        self.renderer.setCamera(self.windowSize, self.Vo[viewIdx,:], -self.VNo[viewIdx,:])
        
        # set light 
        x,y,z = self.renderer.getCameraFrame()
        self.renderer.setLights(x,y,z, np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])) # 3 rendererGB directional lights
        
        # set geometry
        self.renderer.setMesh(V,F)
        
        # rendering
        img = self.renderer.draw()
        
        filtImg = self.filterFunc(img)
        
        
        # compute delta R
        dR = (img - filtImg).reshape(1,3*self.imgSize**2)
        
        # compute dRdN
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
        dRdN = sparse.csc_matrix((data, (row, col)), shape=(3*imgSize**2, 3*F.shape[0]))
        
        # compute dNdV
        dNdV = computedNdV(V,F)
        
        # compute dV
        dV = dR * dRdN * dNdV
        dV = -dV.reshape((dV.shape[1]/3, 3))
        return dV
    def cleanup(self, V,mit):
        # filter gradients eltopo
        if mit%self.elTopoFreq == 0:
            t_eltopo = time.time()
            self.eltopo.update(V)
            V[:], newF = self.eltopo.getMesh()
            firstM = np.zeros(V.shape)
            secondM = np.zeros(V.shape)
            timeStep = 0
            print('eltopo time:', time.time() - t_eltopo)
            gc.collect()
            return True
        return False



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
        opt = self.optimizer(self.gradient
                ,V.shape
                ,cleanupFunc=self.cleanup
                ,**self.optimizer_params)
        V[:] = opt.run(V,maxIter)

        return V,F


