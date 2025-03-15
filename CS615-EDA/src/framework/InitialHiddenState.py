import numpy as np
from .Layer import Layer

class InitialHiddenState(Layer):
    epsilon = 1e-8

    def __init__(self, hiddenSize, xavier_initialization=True):
        super().__init__()

        
        if xavier_initialization:
            self.__hidden = np.zeros(shape = (1, hiddenSize))
        else:
            range = 1e-4
            self.__hidden = np.random.random(size = (1, hiddenSize)) * 2 * range - range
        
        self.__hidden_accumulator = np.zeros(shape = (1, hiddenSize))
        self.len_accumulated = 0
        self.__prevIn = []
        self.__prevOut = []
        
    def forward(self, dataIn=None):
        self.__prevIn.append(dataIn)
        X = self.__hidden
        self.__prevOut.append(X)
        self.len_accumulated += 1
        return X
    
    def gradient(self):
        pass

    def backward(self, gradIn):
        pass
    
    def backward_and_calculateUpdateWeights(self, gradIn, eta):
        prevIn = self.getPrevIn().pop()
        prevOut = self.__prevOut.pop()
        
        N = gradIn.shape[0]
        dJdh = np.sum(gradIn, axis = 0)/N
        self.__hidden_accumulator -= eta * dJdh
        return
    
    def performUpdateWeights(self):
        length = self.len_accumulated
        if length == 0:
            raise ValueError("self.__num_accumulated is zero. You have not accumulated gradients")
        self.__hidden += self.__hidden_accumulator / length
        
        self.__hidden_accumulator = np.zeros(shape = self.__hidden_accumulator.shape)
        self.len_accumulated = 0
        return