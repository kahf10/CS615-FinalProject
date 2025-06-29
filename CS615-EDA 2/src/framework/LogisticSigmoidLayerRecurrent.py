import numpy as np
from .Layer import Layer

class LogisticSigmoidLayerRecurrent(Layer):
    epsilon = 1e-8

    def __init__(self):
        super().__init__()
        self.__prevIn = []
        self.__prevOut = []
        return

    def setPrevIn(self,dataIn):
        self.__prevIn = dataIn
    
    def setPrevOut(self, out):
        self.__prevOut = out
    
    def getPrevIn(self):
        return self.__prevIn
    
    def getPrevOut(self):
        return self.__prevOut

    def forward(self, dataIn):
        self.__prevIn.append(dataIn)
        def logisticSigmoid(x):
            return 1 / (1 + np.exp(-x) + self.epsilon)
        X = logisticSigmoid(dataIn)
        self.__prevOut.append(X)
        return X
    
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        pass

    def backward_and_calculateUpdateWeights(self, gradIn, eta):
        prevIn = self.__prevIn.pop()
        prevOut = self.__prevOut.pop()
        
        # calculateUpdateWeights
        # none
        
        # backward
        sg = prevOut * (1 - prevOut)
        gradOut = np.atleast_2d(gradIn)*sg
        return gradOut
    
    def performUpdateWeights(self):
        return

    def deepCopy(self):
        logit = LogisticSigmoidLayerRecurrent()
        logit.prevIn = self.__prevIn.copy()
        logit.prevOut = self.__prevOut.copy()

        return logit