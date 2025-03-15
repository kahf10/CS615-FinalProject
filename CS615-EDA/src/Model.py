import numpy as np
from Preprocessing import Preprocessing
from framework import *


class Model:
    def __init__(self, filepath, num_epochs=1000, learning_rate=0.01, validation_interval=3, hidden_size=32):
        """
        Initializes the GRU model.

        """
        self.filepath = filepath
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.gru = None
        self.h_prev = None
        self.train_X = None
        self.train_y = None
        self.val_x = None
        self.val_y = None


    def loadData(self):
        preprocessor = Preprocessing(self.filepath)
        train_batch, val_batch = preprocessor.runPipeline()

        self.train_X = train_batch[:-1, :, :]
        self.train_y = train_batch[1:, :, :]
        self.val_X = val_batch[:-1, :, :]
        self.val_y = val_batch[1:, :, :]
        
        self.train_mean = np.mean(self.train_X[:, :, :], axis=(0, 1))
        self.train_std = np.std(self.train_X[:, :, :], ddof=1, axis=(0, 1))
        
        epsilon = 1e-8
        self.train_X = (self.train_X - self.train_mean) / (self.train_std + epsilon)
        self.train_y = (self.train_y - self.train_mean) / (self.train_std + epsilon)
        self.val_X = (self.val_X - self.train_mean) / (self.train_std + epsilon)
        self.val_y = (self.val_y - self.train_mean) / (self.train_std + epsilon)

        print(f"Data Loaded. Train shape {self.train_X.shape} Validation shape {self.val_X.shape} ")

    def initializeModel(self, input_size, batch_size):
        self.gru = GRU(input_size, self.hidden_size)
        self.FC = FullyConnectedLayerRecurrent(self.hidden_size, input_size)
        self.h_initial_layer = InitialHiddenState(batch_size, self.hidden_size)

    def trainModel(self):

        for epoch in range(self.num_epochs):
            h_prev = self.h_initial_layer.forward()

            gradients = []
            loss = 0.0
            for step in range(self.train_X.shape[0]):
                h_prev  = self.gru.forward(self.train_X[step], h_prev)

                output = self.FC.forward(h_prev)
                loss += StaticSquaredError.eval(self.train_y[step], output)
                gradient = StaticSquaredError.gradient(self.train_y[step], output)


                gradients.append(gradient)

            gradIn = np.zeros(shape=self.gru.getPrevOut()[-1].shape)
            for step in range(self.train_X.shape[0] - 1, -1, -1):
                gradIn_to_FC = gradients.pop()
                gradIn_to_gru = self.FC.backward_and_calculateUpdateWeights(gradIn_to_FC, self.learning_rate)
                gradIn += gradIn_to_gru
                _, gradIn = self.gru.backward_and_calculateUpdateWeights(gradIn, self.learning_rate)
            self.h_initial_layer.backward_and_calculateUpdateWeights(gradIn, self.learning_rate)

            self.FC.performUpdateWeights()
            self.gru.performUpdateWeights()
            self.h_initial_layer.performUpdateWeights()

            print(f"Epoch number {epoch+1}, Loss {loss}")

def main():
    model = Model("data\TOS Kaggle data week ending 2022 01 07.csv")
    model.loadData()
    model.initializeModel(16, 800)
    model.trainModel()

if __name__ == "__main__":
    main()







