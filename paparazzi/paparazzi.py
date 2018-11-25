from readOBJ import *
from renderer import *
import gc
import time
import skimage
import skimage.segmentation



class Paparazzi(object):

    def __init__(self,meshPath,offsetPath, imgSize
        # parameters for optimization (NADAM [Dozat et al. 2016])
        ,windowSize = 0.5
        ,timeStep = 0
        ,beta1 = 0.9
        ,beta2 = 0.999
        ,eps = 1e-8
        ,eltopoFreq = 30 # number of iterations to perform El Topo once
            ):
        self.V,self.F = readOBJ(meshPath)
        self.eltopo = ElTopoMesh(self.V,self.F)# build el topo mesh

        # read offset surface 
        self.Vo, self.Fo = readOBJ(offsetPath)
        self.VNo = vertexNormals(self.Vo,self.Fo)
        self.VAo = vertexAreas(self.Vo, self.Fo)
        self.VAoCumsum = np.cumsum(self.VAo) / self.VAo.sum() # for vertex area weighted uniform sampling

        # initialize renderer
        # windowSize = 0.5

        self.windowSize = windowSize
        self.timeStep = timeStep 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.eps = eps 
        self.eltopoFreq = eltopoFreq

    def run(self,maxIter, filterFunc): 
        V = self.V
        F = self.F
        renderer = PaparazziRenderer(imgSize = self.imgSize)
        renderer.setMesh(self.V,self.F)
        firstM = np.zeros(self.V.shape)
        secondM = np.zeros(self.V.shape)
        for iteration in xrange(maxIter):
            # reduce learning rate
            if iteration == 2500:
                lr = 1e-5
        
            # sample camera from offset surface
            viewIdx = np.searchsorted(self.VAoCumsum,np.random.rand())
            renderer.setCamera(windowSize, Vo[viewIdx,:], -VNo[viewIdx,:])
        
            # set light 
            x,y,z = renderer.getCameraFrame()
            renderer.setLights(x,y,z, np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])) # 3 rendererGB directional lights
        
            # set geometry
            renderer.setMesh(V,F)
        
            # rendering
            img = renderer.draw()
        
            filtImg = filterFunc(img)
        
        
            # compute delta R
            dR = (img - filtImg).reshape(1,3*imgSize**2)
        
            # compute dRdN
            idxImg, actFIdx, actRow = renderer.draw("faceIdx")
            LDir, LC = renderer.getLights()
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
        
            # perform NADAM update
            timeStep += 1
            firstM = beta1*firstM + (1-beta1)*dV
            secondM = beta2*secondM + (1-beta2)*(dV**2)
            firstM_cor = firstM / (1-np.power(beta1,timeStep))
            secondM_cor = secondM / (1-np.power(beta2,timeStep))
            newV = V - lr/(np.sqrt(secondM_cor)+eps) * (beta1*firstM_cor + ((1-beta1)*dV/(1-np.power(beta1,timeStep)))) 
        
            print("iteration: %d/%d".format(iteration, maxIter))
        
            # filter gradients eltopo
            if np.mod(iteration, eltopoFreq) == 0:
                t_eltopo = time.time()
                self.eltopo.update(newV)
                newV, newF = self.eltopo.getMesh()
                firstM = np.zeros(newV.shape)
                secondM = np.zeros(newV.shape)
                timeStep = 0
                print('eltopo time:', time.time() - t_eltopo)
                gc.collect()
            V = newV
        renderer.close()
        return V,F


