import numpy as np
from .Layer import Layer

class FullyConnectedLayerRecurrent(Layer):
    epsilon = 1e-8

    def __init__(self, sizeIn, sizeOut, xavier_initialization=True, bias = True):
        super().__init__()
        self.bias = bias
        if xavier_initialization:
            self.__weights = np.random.normal(loc = 0, scale = np.sqrt(6. / (sizeIn + sizeOut)), size = (sizeIn, sizeOut))
            #self.__biases = np.random.normal(loc = 0, scale = np.sqrt(6. / (1 + sizeOut)), size = (1, sizeOut))
            self.__biases = np.zeros(shape = (1, sizeOut))
        else:
            range = 1e-4
            self.__weights = np.random.random(size = (sizeIn, sizeOut)) * 2 * range - range
            self.__biases = np.random.random(size = (1, sizeOut)) * 2 * range - range
        
        self.__weights_accumulator = np.zeros(shape = (sizeIn, sizeOut))
        self.__biases_accumulator = None
        if not bias:
            self.__biases = np.zeros(shape = (1, sizeOut))
            self.__biases_accumulator = np.zeros(shape = (1, sizeOut))
            
        self.len_accumulated = 0
        return

    def getWeights(self):
        return self.__weights
    
    def setWeights(self, weights):
        self.__weights = weights
        return
    
    def getBiases(self):
        if self.bias:
            return self.__biases
        return None
    
    def setBiases(self, biases):
        if self.bias:
            self.__biases = biases
        return

    def forward(self, dataIn):
        self.__prevIn.append(dataIn)
        X = dataIn @ self.__weights + self.__biases
        self.__prevOut.append(X)
        self.len_accumulated += 1
        return X
    
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        pass

    def calculateUpdateWeights(self, gradIn, eta):
        pass
        
    def backward_and_calculateUpdateWeights(self, gradIn, eta):
        prevIn = self.__prevIn.pop()
        prevOut = self.__prevOut.pop()
        
        # calculateUpdateWeights
        N = gradIn.shape[0]
        dJdb = np.sum(gradIn, axis = 0)/N
        dJdW = (prevIn.T @ gradIn)/N
        self.__weights_accumulator -= eta * dJdW
        if self.bias:
            self.__biases_accumulator -= eta * dJdb
            
        # backward
        sg = self.__weights.T
        gradOut = np.atleast_2d(gradIn)@sg
        return gradOut
    
    def performUpdateWeights(self):
        length = self.len_accumulated
        if length == 0:
            raise ValueError("self.__num_accumulated is zero. You have not accumulated gradients")
        self.__weights += self.__weights_accumulator / length
        if self.bias:
            self.__biases += self.__biases_accumulator / length
        
        self.__weights_accumulator = np.zeros(shape = self.__weights_accumulator.shape)
        self.__biases_accumulator = np.zeros(shape = self.__biases_accumulator.shape) if self.bias else None
        self.len_accumulated = 0
        return
