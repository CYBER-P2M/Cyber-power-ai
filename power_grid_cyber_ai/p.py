import pandas as pd
from pathlib import Path

folder = Path("data/raw/uah_triple")

for file_name in ["data1.csv", "data15.csv"]:
    df = pd.read_csv(folder / file_name)

    print(f"\n--- {file_name} ---")
    print("Columns:", df.columns.tolist())

    for column in ["class", "marker", "label"]:
        if column in df.columns:
            print(f"{column} values:")
            print(df[column].value_counts(dropna=False).head(20))