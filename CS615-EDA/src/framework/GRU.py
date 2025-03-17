import numpy as np
from .Layer import Layer
from .FullyConnectedLayerRecurrent import FullyConnectedLayerRecurrent
from .LogisticSigmoidLayerRecurrent import LogisticSigmoidLayerRecurrent
from .TanhLayerRecurrent import TanhLayerRecurrent

class GRU(Layer):
    epsilon = 1e-8

    def __init__(self, inputSizeIn, hiddenSize):
        super().__init__()
        # GRU as defined on Wikipedia: https://en.wikipedia.org/wiki/Gated_recurrent_unit
        
        # update gate vector
        self.z_data_in = FullyConnectedLayerRecurrent(inputSizeIn, hiddenSize)
        self.z_hidden_in = FullyConnectedLayerRecurrent(hiddenSize, hiddenSize, bias=False)
        self.z_activation = LogisticSigmoidLayerRecurrent()
        
        # reset gate vector
        self.r_data_in = FullyConnectedLayerRecurrent(inputSizeIn, hiddenSize)
        self.r_hidden_in = FullyConnectedLayerRecurrent(hiddenSize, hiddenSize, bias=False)
        self.r_activation = LogisticSigmoidLayerRecurrent()
        
        # candidate activation vector
        self.h_hat_data_in = FullyConnectedLayerRecurrent(inputSizeIn, hiddenSize)
        self.h_hat_hidden_in = FullyConnectedLayerRecurrent(hiddenSize, hiddenSize, bias=False)
        self.h_hat_activation = TanhLayerRecurrent()
        
        # output vector
        # no setup
        
        self.__prevIn_hidden = []

        self.__prevIn = []
        self.__prevOut = []
        
    def setPrevIn(self,dataIn):
        self.__prevIn = dataIn
    
    def setPrevOut(self, out):
        self.__prevOut = out
    
    def getPrevIn(self):
        return self.__prevIn
    
    def getPrevOut(self):
        return self.__prevOut
        
    def setPrevIn_hidden(self, dataIn_hidden):
        self.__prevIn_hidden = dataIn_hidden
        
    def getPrevIn_hidden(self):
        return self.__prevIn_hidden
        
    def forward(self, dataIn, hiddenIn):
        self.__prevIn.append(dataIn)
        self.__prevIn_hidden.append(hiddenIn)
        
        # update gate vector
        z_data_in = self.z_data_in.forward(dataIn)
        z_hidden_in = self.z_hidden_in.forward(hiddenIn)
        z_activation = self.z_activation.forward(z_data_in + z_hidden_in)
        
        # reset gate vector
        r_data_in = self.r_data_in.forward(dataIn)
        r_hidden_in = self.r_hidden_in.forward(hiddenIn)
        r_activation = self.r_activation.forward(r_data_in + r_hidden_in)
        
        # candidate activation vector
        h_hat_data_in = self.h_hat_data_in.forward(dataIn)
        h_hat_hidden_in = self.h_hat_hidden_in.forward(r_activation * hiddenIn)
        h_hat_activation = self.h_hat_activation.forward(h_hat_data_in + h_hat_hidden_in)
        
        # output vector
        h_output = (1 - z_activation) * hiddenIn + z_activation * h_hat_activation
        
        self.__prevOut.append(h_output)
        
        return h_output
    
    def gradient(self):
        pass

    def backward(self, gradIn):
        pass
    
    def backward_and_calculateUpdateWeights(self, gradIn, eta):
        prevIn = self.__prevIn.pop()
        h_prevIn = self.__prevIn_hidden.pop()
        h_prevOut = self.__prevOut.pop()

        # Intermediate outputs 
        z = self.z_activation.getPrevOut()[-1]
        r = self.r_activation.getPrevOut()[-1]
        h_hat = self.h_hat_activation.getPrevOut()[-1]

        # GRU output: h = (1 - z) o h_prev + z o h_hat
        dL_dh = gradIn

        # Gradients from the output with respect to z, h_prev, and h_hat:
        dL_dz = dL_dh * (h_hat - h_prevIn)
        dL_dh_hat = dL_dh * z
        dL_dh_prev_out = dL_dh * (1 - z)

        # candidate activation vector
        # h_hat = phi(h_hat_data_in + h_hat_hidden_in)
        grad_h_hat_activation = self.h_hat_activation.backward_and_calculateUpdateWeights(dL_dh_hat, eta)
        grad_data_candidate = self.h_hat_data_in.backward_and_calculateUpdateWeights(grad_h_hat_activation, eta)
        grad_hidden_candidate = self.h_hat_hidden_in.backward_and_calculateUpdateWeights(grad_h_hat_activation, eta)
        # grad_hidden_candidate = r * h_prev, d(r * h_prev)/dr = h_prev  and  d(r * h_prev)/dh_prev = r.
        dL_dr_candidate = grad_hidden_candidate * h_prevIn
        dL_dh_candidate = grad_hidden_candidate * r
        
        # reset gate vector
        # The gradient dL/dr comes solely from the candidate branch:
        delta_r = self.r_activation.backward_and_calculateUpdateWeights(dL_dr_candidate, eta)
        grad_data_r   = self.r_data_in.backward_and_calculateUpdateWeights(delta_r, eta)
        grad_hidden_r = self.r_hidden_in.backward_and_calculateUpdateWeights(delta_r, eta)

        # update gate vector
        # The input to z_activation was the sum: z_data_in + z_hidden_in.
        delta_z = self.z_activation.backward_and_calculateUpdateWeights(dL_dz, eta)
        grad_data_z   = self.z_data_in.backward_and_calculateUpdateWeights(delta_z, eta)
        grad_hidden_z = self.z_hidden_in.backward_and_calculateUpdateWeights(delta_z, eta)

        # gradient for input
        grad_x = grad_data_candidate + grad_data_r + grad_data_z

        # gradient for h_prev
        grad_h_prev = dL_dh_prev_out + dL_dh_candidate + grad_hidden_r + grad_hidden_z

        return grad_x, grad_h_prev

    def performUpdateWeights(self, learning_rate=0.01):
        # Adaptive Learning Rate Decay (Optional)
        # Update gate vector
        self.z_data_in.performUpdateWeights(learning_rate)
        self.z_hidden_in.performUpdateWeights(learning_rate)
        self.z_activation.performUpdateWeights()

        # Reset gate vector
        self.r_data_in.performUpdateWeights(learning_rate)
        self.r_hidden_in.performUpdateWeights(learning_rate)
        self.r_activation.performUpdateWeights()

        # Candidate activation vector
        self.h_hat_data_in.performUpdateWeights(learning_rate)
        self.h_hat_hidden_in.performUpdateWeights(learning_rate)
        self.h_hat_activation.performUpdateWeights()

        return

