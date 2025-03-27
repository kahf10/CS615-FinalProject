import os
import pandas as pd

class EDA:
    def __init__(self):
        self.df = None

    def readFile(self, filePath):
        self.df = pd.read_csv(filePath)

    def printSummary(self):
        print(self.df.columns)
        print(self.df.describe())
        print(self.df.head())
        print(self.df.shape[1])

def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    eda = EDA()

    # Dynamically find absolute path to `data/` directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

    # List of dataset files
    files = [
        "TOS Kaggle data week ending 2021 01 01.csv",
        "TOS Kaggle data week ending 2022 01 07.csv",
        "TOS Kaggle data week ending 2023 01 06.csv",
        "TOS Kaggle data week ending 2024 01 05.csv",
        "TOS Kaggle data week ending 2025 01 03.csv",
    ]

    # Iterate through the dataset files
    for file in files:
        file_path = os.path.join(base_dir, file)  # Construct absolute path
        understandDatasetSummaryForEachYear(eda, file.split()[-1], file_path)

def understandDatasetSummaryForEachYear(eda, year, filepath):
    print(f"'{year}' Data Summary")
    eda.readFile(filepath)
    eda.printSummary()
    print(80 * '-')

if __name__ == '__main__':
    main()
