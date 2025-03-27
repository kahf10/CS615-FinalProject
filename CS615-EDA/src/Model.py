import numpy as np
import matplotlib.pyplot as plt
import time

from Preprocessing import Preprocessing
from framework import *

# Global Loss History (To Be Updated in Training)
loss_history = []

class Model:
    def __init__(self, filepath, num_epochs=100, learning_rate=0.01, period=[1, 3, 5], hidden_size=20, random_seed=42):
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
        self.val_X = None
        self.val_y = None
        self.period = period
        self.num_models = len(period)
        self.prediction_steps = 50
        self.random_seed = random_seed

    @staticmethod
    def average(arr):
        return sum(arr) / len(arr)

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

    def initializeModel(self, input_size):
        self.gru = [GRU(input_size, self.hidden_size) for _ in range(self.num_models)]
        self.FC = [FullyConnectedLayerRecurrent(self.hidden_size, input_size) for _ in range(self.num_models)]
        self.h_initial_layer = [InitialHiddenState(self.hidden_size) for _ in range(self.num_models)]

    def trainModel(self):


        prev_loss = float('inf')
        min_learning_rate = 1e-6
        loss_threshold = 0.025
        decay_rate = 0.995

        for epoch in range(self.num_epochs):
            batch_size = self.train_X[0].shape[0]
            h_prev = [h_initial_layer.forward(batch_size) for h_initial_layer in self.h_initial_layer]
            output = [None for _ in range(self.num_models)]

            gradients = []
            loss = 0.0
            for step in range(self.train_X.shape[0]):
                for i in range(self.num_models):
                    if step % self.period[i] == 0:
                        h_prev[i] = self.gru[i].forward(self.train_X[step], h_prev[i])
                        output[i] = self.FC[i].forward(h_prev[i])
                total_output = Model.average(output)

                loss += StaticSquaredError.eval(self.train_y[step], total_output)
                gradient = StaticSquaredError.gradient(self.train_y[step], total_output)
                  
                gradients.append(gradient)

            loss_history.append(loss)

            # delta_loss = abs(prev_loss- loss)
            # if delta_loss < loss_threshold:
            #     self.learning_rate = max(self.learning_rate * decay_rate, min_learning_rate)
            # prev_loss = loss

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
                self.FC[i].performUpdateWeights(self.learning_rate)
                self.gru[i].performUpdateWeights()
                self.h_initial_layer[i].performUpdateWeights()

            print(f"Epoch number {epoch+1}, Loss {loss}")

    def validateModel(self):

        batch_size = self.val_X[0].shape[0]
        h_prev = [h_initial_layer.forward(batch_size) for h_initial_layer in self.h_initial_layer]
        output = [None for _ in range(self.num_models)]

        # Rows show how many time steps we are predicting into the future
        # Columns show how many time steps we are using to make the prediction
        x_predictions = [[] for _ in range(self.prediction_steps)]
        y_predictions = [[] for _ in range(self.prediction_steps)]


        for step in range(self.val_X.shape[0] - self.prediction_steps):
            dataIn = self.val_X[step]
            storedModelGru = [self.gru[i].deepCopy() for i in range(self.num_models)]
            storedModelFC = [self.FC[i].deepCopy() for i in range(self.num_models)]
            for j in range(self.prediction_steps):
                for i in range(self.num_models):
                    if step % self.period[i] == 0:
                        h_prev[i] = self.gru[i].forward(dataIn, h_prev[i])
                        output[i] = self.FC[i].forward(h_prev[i])
                total_output = Model.average(output)
                real_output = self.val_y[step + j]

                x_predictions[j].append(total_output)
                y_predictions[j].append(real_output)
                dataIn = total_output
            for i in range(self.num_models):
                self.gru[i].restore(storedModelGru[i])
                self.FC[i].restore(storedModelFC[i])

        return x_predictions, y_predictions

def main():
    model = Model("./data/Jan2022Data.csv")
    model.loadData()
    model.initializeModel(16)
    model.trainModel()
    x_preds, y_preds = model.validateModel()

    computeMape(x_preds, y_preds)
    plotScatterPredictions(x_preds, y_preds)


def computeMape(x_preds, y_preds):
    for step in range(len(x_preds)):
        x_step = np.array(x_preds[step])  # Shape: (99, 160, 16)
        y_step = np.array(y_preds[step])  # Shape: (99, 160, 16)

        # x_step = normalizeStep(x_step)
        # y_step = normalizeStep(y_step)
        mask = y_step != 0

        mape_per_step = np.mean(np.abs((y_step[mask] - x_step[mask]) / y_step[mask])) * 100
        print(f"Step {step + 1}: MAPE = {mape_per_step:.2f}%")


def normalizeStep(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    return (data - mean) / std


def plotScatterPredictions(x_preds, y_preds):
    x_preds = np.array(x_preds)  # (50, 99, 160, 16)
    y_preds = np.array(y_preds)  # (50, 99, 160, 16)

    x_preds = np.mean(x_preds, axis=-1)  # (50, 99, 160)
    y_preds = np.mean(y_preds, axis=-1)  # (50, 99, 160)

    x_flat = x_preds.flatten()  # (50 * 99 * 160,)
    y_flat = y_preds.flatten()  # (50 * 99 * 160,)

    num_steps, num_start_times, num_samples = x_preds.shape  # (50, 99, 160)
    timesteps = np.repeat(np.arange(num_steps), num_start_times * num_samples)  # (50 * 99 * 160,)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(y_flat, x_flat, c=timesteps, cmap="viridis", alpha=0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Prediction Step")

    plt.plot([y_flat.min(), y_flat.max()], [y_flat.min(), y_flat.max()], linestyle="--", color="red", label="Perfect Predictions")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Scatter Plot:")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()







