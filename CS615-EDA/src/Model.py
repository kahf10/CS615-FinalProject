import numpy as np
from Preprocessing import Preprocessing
from framework import *


class Model:
    def __init__(self, filepath, num_epochs=1000, learning_rate=0.01, period=[1, 3, 5, 7, 9, 11], hidden_size=20):
        """
        Initializes the GRU model.

        """
        self.filepath = filepath
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.gru = []
        self.FC = []
        self.h_prev = None
        self.train_X = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.period = period
        self.num_models = len(period)

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
        self.gru = [GRU(input_size, self.hidden_size) for _ in range(self.num_models)]
        self.FC = [FullyConnectedLayerRecurrent(self.hidden_size, input_size) for _ in range(self.num_models)]
        self.h_initial_layer = [InitialHiddenState(batch_size, self.hidden_size) for _ in range(self.num_models)]

    def trainModel(self):
        def average(arr):
            return sum(arr) / len(arr)
        for epoch in range(self.num_epochs):
            h_prev = [h_initial_layer.forward() for h_initial_layer in self.h_initial_layer]
            output = [None for _ in range(self.num_models)]

            gradients = []
            loss = 0.0
            for step in range(self.train_X.shape[0]):
                for i in range(self.num_models):
                    if step % self.period[i] == 0:
                        h_prev[i] = self.gru[i].forward(self.train_X[step], h_prev[i])
                        output[i] = self.FC[i].forward(h_prev[i])
                total_output = average(output)

                loss += StaticSquaredError.eval(self.train_y[step], total_output)
                gradient = StaticSquaredError.gradient(self.train_y[step], total_output)
                  
                gradients.append(gradient)

            gradIn = [np.zeros(shape=gru.getPrevOut()[-1].shape) for gru in self.gru]
            gradIn_to_FC = [np.zeros(shape=gradients[-1].shape) for _ in range(self.num_models)]
            for step in range(self.train_X.shape[0] - 1, -1, -1):
                gradient = gradients.pop() / self.num_models
                for i in range(self.num_models):
                    gradIn_to_FC[i] += gradient
                    if step % self.period[i] == 0:
                        gradIn_to_gru = self.FC[i].backward_and_calculateUpdateWeights(gradIn_to_FC[i], self.learning_rate)
                        gradIn[i] += gradIn_to_gru
                        _, gradIn[i] = self.gru[i].backward_and_calculateUpdateWeights(gradIn[i], self.learning_rate)
                        
                        gradIn_to_FC[i] = np.zeros(shape=gradIn_to_FC[i].shape)
            for i, grad in enumerate(gradIn):
                self.h_initial_layer[i].backward_and_calculateUpdateWeights(grad, self.learning_rate)

            for i in range(self.num_models):
                self.FC[i].performUpdateWeights()
                self.gru[i].performUpdateWeights()
                self.h_initial_layer[i].performUpdateWeights()

            print(f"Epoch number {epoch+1}, Loss {loss}")

def main():
    model = Model("data\TOS Kaggle data week ending 2022 01 07.csv")
    model.loadData()
    model.initializeModel(16, 800)
    model.trainModel()

if __name__ == "__main__":
    main()







