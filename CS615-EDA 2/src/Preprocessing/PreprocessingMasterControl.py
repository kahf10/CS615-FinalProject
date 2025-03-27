from Preprocessing import Preprocessing

class PreprocessingMasterControl:
    def __init__(self, filepath):
        """
        Initializes the master control for preprocessing.

        Args:
        - filepath (str): Path to the dataset CSV file.
        """
        self.filepath = filepath

    def run(self):
        """
        Runs the full preprocessing pipeline.
        """
        preprocessor = Preprocessing(self.filepath)
        train_batch, val_batch = preprocessor.runPipeline()
        return train_batch, val_batch

# Example usage
if __name__ == "__main__":
    filepath = "../../data/TOS Kaggle data week ending 2022 01 07.csv"  # Modify as needed
    master = PreprocessingMasterControl(filepath)
    train_batch, val_batch = master.run()