import numpy as np
import pandas as pd


class Preprocessing:
    def __init__(self, filepath, num_samples=800, events_per_sample=150, feature_count=16, interval=3, train_split=0.8):
        """
        Initializes the preprocessing class.

        Args:
        - filepath (str): Path to the CSV file.
        - num_samples (int): Number of samples in the dataset.
        - events_per_sample (int): Number of events per sample (time intervals per sequence).
        - feature_count (int): Number of stock price features per event.
        - interval (int): Time interval between events in seconds.
        - train_split (float): Percentage of data to be used for training (default=80%).
        """
        self.filepath = filepath
        self.num_samples = num_samples
        self.events_per_sample = events_per_sample
        self.feature_count = feature_count  # Now correctly set to 16
        self.interval = interval
        self.train_split = train_split
        self.data = None
        self.tensor = None
        self.train_batches = None
        self.val_batches = None

    # 1. Load & Trim Data
    def loadData(self):
        """Loads only the required number of rows."""
        total_required_rows = self.num_samples * self.events_per_sample

        # Load only the necessary rows
        df = pd.read_csv(self.filepath, nrows=total_required_rows)

        self.data = df
        print(f"Loaded dataset with shape: {df.shape} (Trimmed to {total_required_rows} rows)")

    # 2. Remove Unnecessary Columns
    def removeColumns(self):
        """
        Removes non-relevant columns like technical indicators, volume. And ID and TimeStamp.
        """
        columns_to_remove = ['ID', 'TimeStamp', '/ES SMA20', '/ES SMA50', '/ES volume', 'TLT volume']

        self.data = self.data.drop(columns=columns_to_remove, errors='ignore')
        print(f"Removed unnecessary columns. New shape: {self.data.shape}")

    # 3. Reshape Data into 3D Tensor
    def reshapeToTensor(self):
        """
        Converts the processed dataframe into a 3D tensor of shape:
        (events_per_sample, num_samples, feature_count)
        """
        # Convert DataFrame to NumPy array
        data_array = self.data.to_numpy()

        # # Debugging data size
        # expected_size = self.num_samples * self.events_per_sample * self.feature_count
        # actual_size = data_array.size
        #
        # if actual_size != expected_size:
        #     raise ValueError(
        #         f"Data size mismatch! Expected {expected_size} elements, but got {actual_size}."
        #     )

        # Reshape into (events, samples, features)
        self.tensor = data_array.reshape(
            (self.events_per_sample, self.num_samples, self.feature_count)
        )

        print(f"Reshaped data into tensor of shape: {self.tensor.shape}")

    # 4. Create batches
    def createBatches(self):
        """
        Creates a single full batch for training and validation.

        Returns:
        - train_batch: Shape (events_per_sample, train_size, feature_count)
        - val_batch: Shape (events_per_sample, val_size, feature_count)
        """
        # Shuffle indices before selecting training & validation samples
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)

        # Split into training and validation sets
        split_idx = int(self.num_samples * self.train_split)
        train_indices = indices[:split_idx]  # 80% of samples
        val_indices = indices[split_idx:]  # 20% of samples

        # Full batch training: Use all selected training samples in one batch
        train_batch = self.tensor[:, train_indices, :]
        val_batch = self.tensor[:, val_indices, :]

        print(f"Generated 1 full training batch of shape {train_batch.shape} and 1 validation batch of shape {val_batch.shape}.")

        return train_batch, val_batch

    def runPipeline(self):
        """
        Runs the full preprocessing pipeline.

        Returns:
        - Training and validation batches.
        """
        self.loadData()
        self.removeColumns()
        self.reshapeToTensor()
        return self.createBatches()  # Returns train & val batches
