import numpy as np
import numpy.linalg


class Optimizer(object):
    def run(self,V,maxIter=1000):
        myV = V.copy()
        print("Starting iterative optimization")
        for mit in xrange(maxIter):
            print("iteration: {}/{}".format(mit, maxIter))
            if self.__step__(myV,mit):
                break
            else:
                ret,myV = self.cleanupFunc(myV,mit)

        return myV


    def __init__(self, gradFunc, cleanupFunc):
        self.gradFunc = gradFunc
        self.cleanupFunc = cleanupFunc

    def __step__(self,V,mit):
        raise NotImplementedError()




class GDOptimizer(Optimizer):
    def __init__(self
            ,gradFunc
            ,cleanupFunc = lambda i,x:(False,x)
            ,timestep=1e-2
            ,eps = 1e-8
            ):
        super(GDOptimizer,self).__init__(gradFunc,cleanupFunc)
        self.timestep = timestep
        self.eps = 1e-8

    def __step__(self,V,mit):

        dV = self.gradFunc(V)
        V[...] = V - self.timestep * dV
        return np.linalg.norm(V) < self.eps



class NADAMOptimizer(Optimizer):
    def __init__(self
            ,gradFunc
            ,cleanupFunc = lambda i,x:(False,x)
            ,learning_rates= {0:1e-5}
            ,beta1 = 0.9
            ,beta2 = 0.999
            ,eps = 1e-8
            ):
        super(NADAMOptimizer,self).__init__(gradFunc,cleanupFunc)
        
        if type(learning_rates) is dict:
            self.learning_rates = learning_rates
        else:
            self.learning_rates = {0,learning_rates}
        self.learning_rate = self.learning_rates[0]



        self.beta1 = beta1 
        self.beta2 = beta2 
        self.eps = eps 

        self.relative_iter = 0


    def run(self,V,maxIter=1000):
        myV = V.copy()
        print("Starting NADAM optimization")
        for mit in xrange(maxIter):
            print("iteration: {}/{}".format(mit, maxIter))
            if self.__step__(myV,mit):
                break
            else:
                ret,myV = self.cleanupFunc(myV,mit)
                if ret:
                    #reset shape
                    self.relative_iter = 0


    def __reset_state__(self,V):
        self.firstM = np.zeros(V.shape)
        self.secondM = np.zeros(V.shape)
        self.relative_iter = 0

    def __step__(self,V,mit):
        if self.relative_iter == 0:
            self.__reset_state__(V)

        self.relative_iter += 1
        relative_it = self.relative_iter
        #update learning rates
        if mit in self.learning_rates:
            self.learning_rate = self.learning_rates[mit]

        #aliasing for convenience 
        beta1 = self.beta1
        beta2 = self.beta2
        lr = self.learning_rate
        
        dV = self.gradFunc(V)
        #update 
        self.firstM = beta1*self.firstM + (1-beta1)*dV
        self.secondM = beta2*self.secondM + (1-beta2)*(dV**2)
        #remove bias
        firstM_cor = self.firstM / (1-beta1**relative_it)
        secondM_cor = self.secondM / (1-beta2**relative_it)
        #return the update
        V[...] = V - lr/(np.sqrt(secondM_cor)+self.eps) * (beta1*firstM_cor + ((1-beta1)*dV/(1-beta1**relative_it))) 
        return False



