import numpy as np
from .Layer import Layer

class TanhLayerRecurrent(Layer):
    epsilon = 1e-8

    def __init__(self):
        super().__init__()
        self.__prevIn = []
        self.__prevOut = []
        return

    def forward(self, dataIn):
        self.__prevIn.append(dataIn)
        def tanh(x):
            pos = np.exp(x)
            neg = np.exp(-x)
            numerator = pos - neg
            denominator = pos + neg
            return numerator / (denominator + self.epsilon)
        X = tanh(dataIn)
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
        sg = 1 - np.square(prevOut)
        gradOut = np.atleast_2d(gradIn)*sg
        return gradOut
    
    def performUpdateWeights(self):
        return