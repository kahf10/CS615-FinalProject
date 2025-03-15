from Preprocessing import Preprocessing
from framework import *


class Model:
    def __init__(self, filepath, num_epochs=100, learning_rate=0.01, validation_interval=3, hidden_size=20):
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

        print(f"Data Loaded. Train shape {self.train_X.shape} Validation shape {self.val_X.shape} ")

    def initializeModel(self, input_size):
        self.gru = GRU(input_size, self.hidden_size)
        self.FC = FullyConnectedLayerRecurrent(self.hidden_size, input_size)
        self.h_initial_layer = InitialHiddenState(self.hidden_size)

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

            gradIn = None
            for step in range(self.train_X.shape[0] - 1, -1, -1):
                gradIn = gradients.pop()
                gradIn = self.FC.backward_and_calculateUpdateWeights(gradIn, self.learning_rate)
                gradIn = self.gru.backward_and_calculateUpdateWeights(gradIn, self.learning_rate)
            self.h_initial_layer.backward_and_calculateUpdateWeights(gradIn, self.learning_rate)

            self.FC.performUpdateWeights()
            self.gru.performUpdateWeights()
            self.h_initial_layer.performUpdateWeights()

            print(f"Epoch number {epoch}, Loss {loss}")

def main():
    model = Model("/Users/kahfhussain/Desktop/CS615-FinalProject/CS615-EDA/data/TOS Kaggle data week ending 2022 01 07.csv")
    model.loadData()
    model.initializeModel(16)
    model.trainModel()

if __name__ == "__main__":
    main()







