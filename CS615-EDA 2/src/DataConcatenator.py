import os
import pandas as pd

data_dir = "../data"

csv_files = [file for file in os.listdir(data_dir) if file.endswith(".csv")]
jan_files = sorted([file for file in csv_files if "2022 01" in file])

dataframes = []

for file in jan_files:
    file_path = os.path.join(data_dir, file)
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path)
    dataframes.append(df)

if dataframes:
    merged_df = pd.concat(dataframes, ignore_index=False)

    output_file = os.path.join(data_dir, "Jan2022Data.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as {output_file}")
else:
    print("No files found")