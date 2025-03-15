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

    understandDatasetSummaryForEachYear(eda, 2021, '../data/TOS Kaggle data week ending 2021 01 01.csv')

    understandDatasetSummaryForEachYear(eda, 2022, '../data/TOS Kaggle data week ending 2022 01 07.csv')

    understandDatasetSummaryForEachYear(eda, 2023, '../data/TOS Kaggle data week ending 2023 01 06.csv')

    understandDatasetSummaryForEachYear(eda, 2024, '../data/TOS Kaggle data week ending 2024 01 05.csv')

    understandDatasetSummaryForEachYear(eda, 2025, '../data/TOS Kaggle data week ending 2025 01 03.csv')


def understandDatasetSummaryForEachYear(eda, year, filepath):
    print(f"'{year}' Data Summary")
    eda.readFile(filepath)
    eda.printSummary()
    print(80 * '-')


if __name__ == '__main__':
    main()