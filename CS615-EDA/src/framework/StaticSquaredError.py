import numpy as np

class StaticSquaredError():
    
    @staticmethod
    def eval(Y, Yhat):
        return np.mean(np.square(Y - Yhat))

    @staticmethod
    def gradient(Y, Yhat):
        return -2 * (Y - Yhat)